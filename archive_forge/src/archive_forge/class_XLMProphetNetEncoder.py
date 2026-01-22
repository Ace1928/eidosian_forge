import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
@add_start_docstrings('The standalone encoder part of the XLMProphetNetModel.', XLM_PROPHETNET_START_DOCSTRING)
class XLMProphetNetEncoder(XLMProphetNetPreTrainedModel):
    """
    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`XLMProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    """

    def __init__(self, config: XLMProphetNetConfig, word_embeddings: nn.Embedding=None):
        super().__init__(config)
        self.word_embeddings = word_embeddings if word_embeddings is not None else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = XLMProphetNetPositionalEmbeddings(config)
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList([XLMProphetNetEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, XLMProphetNetEncoder
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
        >>> model = XLMProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone")
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None and inputs_embeds is None:
            raise ValueError('Either input_ids or inputs_embeds has to be passed.')
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError('Make sure to only pass input_ids or inputs_embeds.')
        elif input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :].repeat(1, self.config.num_encoder_attention_heads, 1, 1)) * torch.finfo(self.dtype).min
            extended_attention_mask = extended_attention_mask.to(inputs_embeds.dtype)
        else:
            extended_attention_mask = None
        position_embeddings, position_ids = self.position_embeddings(inputs_embeds.shape[:2], inputs_embeds.device)
        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.embeddings_layer_norm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.config.dropout, training=self.training)
        encoder_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_hidden_states = encoder_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, extended_attention_mask, head_mask[idx] if head_mask is not None else None, output_attentions)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask=extended_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_hidden_states = encoder_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_hidden_states, attentions=all_attentions)