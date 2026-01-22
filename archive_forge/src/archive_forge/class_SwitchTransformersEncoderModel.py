import copy
import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_switch_transformers import SwitchTransformersConfig
@add_start_docstrings("The bare SWITCH_TRANSFORMERS Model transformer outputting encoder's raw hidden-states without any specific head on top.", SWITCH_TRANSFORMERS_START_DOCSTRING)
class SwitchTransformersEncoderModel(SwitchTransformersPreTrainedModel):
    _tied_weights_keys = ['encoder.embed_tokens.weight']

    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SwitchTransformersStack(encoder_config, self.shared)
        self.post_init()
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SWITCH_TRANSFORMERS_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_router_logits: Optional[bool]=True, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], MoEModelOutput]:
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, SwitchTransformersEncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
        >>> model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_router_logits=output_router_logits, return_dict=return_dict)
        return encoder_outputs