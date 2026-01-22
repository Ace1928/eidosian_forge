import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from .configuration_mamba import MambaConfig
from ..deprecated._archive_maps import MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
@add_start_docstrings('The bare MAMBA Model transformer outputting raw hidden-states without any specific head on top.', MAMBA_START_DOCSTRING)
class MambaModel(MambaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if 'embedding.' in k:
                state_dict[k.replace('embedding.', 'embeddings.')] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MambaOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, cache_params: Optional[MambaCache]=None, use_cache: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, MambaOutput]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache if not self.training else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one')
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        if cache_params is None and use_cache:
            cache_params = MambaCache(self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]
        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, cache_params, all_hidden_states] if v is not None))
        return MambaOutput(last_hidden_state=hidden_states, cache_params=cache_params if use_cache else None, hidden_states=all_hidden_states)