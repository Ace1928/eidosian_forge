import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn, tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ernie_m import ErnieMConfig
@add_start_docstrings('The bare ErnieM Model transformer outputting raw hidden-states without any specific head on top.', ERNIE_M_START_DOCSTRING)
class ErnieMModel(ErnieMPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super(ErnieMModel, self).__init__(config)
        self.initializer_range = config.initializer_range
        self.embeddings = ErnieMEmbeddings(config)
        self.encoder = ErnieMEncoder(config)
        self.pooler = ErnieMPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layers[layer].self_attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(processor_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[tensor]=None, position_ids: Optional[tensor]=None, attention_mask: Optional[tensor]=None, head_mask: Optional[tensor]=None, inputs_embeds: Optional[tensor]=None, past_key_values: Optional[Tuple[Tuple[tensor]]]=None, use_cache: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time.')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        if attention_mask is None:
            attention_mask = (input_ids == self.config.pad_token_id).to(torch.float32)
            attention_mask *= torch.finfo(attention_mask.dtype).min
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = torch.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = torch.concat([past_mask, attention_mask], dim=-1)
        elif attention_mask.ndim == 2:
            attention_mask = attention_mask.to(torch.float32)
            attention_mask = 1.0 - attention_mask
            attention_mask *= torch.finfo(attention_mask.dtype).min
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, past_key_values=past_key_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            sequence_output = encoder_outputs[0]
            pooler_output = self.pooler(sequence_output) if self.pooler is not None else None
            return (sequence_output, pooler_output) + encoder_outputs[1:]
        sequence_output = encoder_outputs['last_hidden_state']
        pooler_output = self.pooler(sequence_output) if self.pooler is not None else None
        hidden_states = None if not output_hidden_states else encoder_outputs['hidden_states']
        attentions = None if not output_attentions else encoder_outputs['attentions']
        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence_output, pooler_output=pooler_output, hidden_states=hidden_states, attentions=attentions)