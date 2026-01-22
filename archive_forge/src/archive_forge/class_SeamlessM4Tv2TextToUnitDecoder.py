import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
@add_start_docstrings('Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`SeamlessM4Tv2DecoderLayer`].', SEAMLESS_M4T_V2_START_DOCSTRING, '\n        embed_tokens (`nn.Embedding`, *optional*):\n            Input embedding\n    ')
class SeamlessM4Tv2TextToUnitDecoder(SeamlessM4Tv2PreTrainedModel):

    def __init__(self, config: SeamlessM4Tv2Config, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_char = nn.Embedding(config.char_vocab_size, config.hidden_size)
        self.embed_char_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(self.max_target_positions, config.hidden_size, padding_idx=self.padding_idx)
        self.pos_emb_alpha_char = nn.Parameter(torch.ones(1))
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.duration_predictor = SeamlessM4Tv2VariancePredictor(config.variance_predictor_embed_dim, config.variance_predictor_hidden_dim, config.variance_predictor_kernel_size, config.variance_pred_dropout)
        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(self.max_target_positions, config.hidden_size, padding_idx=self.padding_idx)
        layers = []
        for _ in range(config.decoder_layers):
            layers.append(SeamlessM4Tv2TextToUnitDecoderLayer(config, decoder_attention_heads=config.decoder_attention_heads, decoder_ffn_dim=config.decoder_ffn_dim))
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, char_input_ids: torch.LongTensor=None, char_count_per_id: torch.LongTensor=None, encoder_hidden_states: torch.FloatTensor=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, SeamlessM4Tv2TextToUnitDecoderOutput]:
        """
        Args:
            char_input_ids (`torch.LongTensor` of shape `(batch_size, char_sequence_length)`):
                Character indices. The correspondence between characters and indices can be found in `char_to_id`, a
                dictionary in the generation configuration.
            char_count_per_id (`torch.Tensor` of shape `(batch_size, encoder_sequence_length)`):
                Number of characters per text input id.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
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
        char_padding_mask = _compute_new_attention_mask(char_input_ids, char_count_per_id.sum(1))
        char_hidden_states = self._hard_upsample(encoder_hidden_states, char_count_per_id)
        char_positions = self.pos_emb_alpha_char * self.embed_char_positions(inputs_embeds=char_hidden_states)
        char_hidden_states = self.embed_char(char_input_ids) * self.embed_scale + char_positions + char_hidden_states
        log_dur_pred = self.duration_predictor(char_hidden_states, padding_mask=char_padding_mask)
        dur_out = torch.clamp(torch.round(torch.exp(log_dur_pred) - 1).long(), min=1)
        dur_out = dur_out.masked_fill(~char_padding_mask.bool(), 0.0)
        char_hidden_states = self._hard_upsample(char_hidden_states, dur_out)
        positions = self.pos_emb_alpha * self.embed_positions(inputs_embeds=char_hidden_states)
        hidden_states = char_hidden_states + positions
        padding_mask = _compute_new_attention_mask(hidden_states, dur_out.sum(1))
        attention_mask = _prepare_4d_attention_mask(padding_mask, hidden_states.dtype)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask, padding_mask, output_attentions)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, padding_mask=padding_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[2],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attns, padding_mask] if v is not None))
        return SeamlessM4Tv2TextToUnitDecoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attns, padding_mask=padding_mask)