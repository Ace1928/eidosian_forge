import copy
import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_bigbird_pegasus import BigBirdPegasusConfig
class BigBirdPegasusEncoder(BigBirdPegasusPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BigBirdPegasusEncoderLayer`].

    Args:
        config: BigBirdPegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BigBirdPegasusConfig, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__(config)
        self.attention_type = config.attention_type
        self.block_size = config.block_size
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.embed_positions = BigBirdPegasusLearnedPositionalEmbedding(config.max_position_embeddings, embed_dim)
        self.layers = nn.ModuleList([BigBirdPegasusEncoderLayer(config, seed=i) for i in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=hidden_states.device)
        attention_mask = attention_mask.long()
        max_tokens_to_attend = (5 + 2 * self.config.num_random_blocks) * self.config.block_size
        if self.attention_type == 'block_sparse' and input_shape[1] <= max_tokens_to_attend:
            sequence_length = input_shape[1]
            logger.warning(f"Attention type 'block_sparse' is not possible if sequence_length: {sequence_length} <= num global tokens: 2 * config.block_size + min. num sliding tokens: 3 * config.block_size + config.num_random_blocks * config.block_size + additional buffer: config.num_random_blocks * config.block_size = {max_tokens_to_attend} with config.block_size = {self.config.block_size}, config.num_random_blocks = {self.config.num_random_blocks}. Changing attention type to 'original_full'...")
            self.set_attention_type('original_full')
        if self.attention_type == 'block_sparse':
            padding_len, hidden_states, attention_mask = self._pad_to_block_size(hidden_states, attention_mask)
        else:
            padding_len = 0
        if self.attention_type == 'original_full':
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
            blocked_encoder_mask = band_mask = from_mask = to_mask = None
        elif self.attention_type == 'block_sparse':
            blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(attention_mask, self.block_size)
            attention_mask = None
        else:
            raise ValueError(f'attention_type can either be original_full or block_sparse, but is {self.attention_type}')
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, band_mask, from_mask, to_mask, blocked_encoder_mask, blocked_encoder_mask, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, band_mask=band_mask, from_mask=from_mask, to_mask=to_mask, from_blocked_mask=blocked_encoder_mask, to_blocked_mask=blocked_encoder_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        hidden_states = self.layernorm_embedding(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if padding_len > 0:
            hidden_states = hidden_states[:, :-padding_len]
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        self.encoder_o = hidden_states
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

    def set_attention_type(self, value: str):
        if value not in ['original_full', 'block_sparse']:
            raise ValueError(f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}")
        if value == self.attention_type:
            return
        self.attention_type = value
        for layer in self.layers:
            layer.set_attention_type(value)

    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
        batch_size, seq_length = attention_mask.size()
        if seq_length % block_size != 0:
            raise ValueError(f'Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}.')

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            exp_blocked_to_pad = torch.cat([to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2)
            band_mask = torch.einsum('blq,blk->blqk', from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask
        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)
        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)
        return (blocked_encoder_mask, band_mask, from_mask, to_mask)

    def _pad_to_block_size(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention."""
        block_size = self.config.block_size
        batch_size, seq_len = hidden_states.shape[:2]
        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            logger.warning_once(f'Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of `config.block_size`: {block_size}')
            pad_id = self.config.pad_token_id
            device = hidden_states.device
            input_ids_padding = torch.ones((batch_size, padding_len), dtype=torch.long, device=device) * pad_id
            inputs_embeds_padding = self.embed_tokens(input_ids_padding)
            hidden_states = torch.cat([hidden_states, inputs_embeds_padding], dim=-2)
            attention_mask = nn.functional.pad(attention_mask, (0, padding_len), value=0)
        return (padding_len, hidden_states, attention_mask)