import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_rwkv import RwkvConfig
@add_start_docstrings('The bare RWKV Model transformer outputting raw hidden-states without any specific head on top.', RWKV_START_DOCSTRING)
class RwkvModel(RwkvPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([RwkvBlock(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)
        self.layers_are_rescaled = False
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=RwkvOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, state: Optional[List[torch.FloatTensor]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RwkvOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache if not self.training else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.training == self.layers_are_rescaled:
            self._rescale_layers()
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is None and inputs_embeds is None:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if use_cache and state is None:
            shape = (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers)
            state = [torch.zeros(*shape, dtype=inputs_embeds.dtype if i <= 1 else torch.float32, device=inputs_embeds.device) for i in range(5)]
            state[4] -= 1e+30
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        hidden_states = inputs_embeds
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for idx, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                hidden_states, state, attentions = self._gradient_checkpointing_func(block.__call__, hidden_states, state, use_cache, output_attentions)
            else:
                hidden_states, state, attentions = block(hidden_states, state=state, use_cache=use_cache, output_attentions=output_attentions)
            if self.layers_are_rescaled and self.config.rescale_every > 0 and ((idx + 1) % self.config.rescale_every == 0):
                hidden_states = hidden_states / 2
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)
        hidden_states = self.ln_out(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((x for x in [hidden_states, state, all_hidden_states, all_self_attentions] if x is not None))
        return RwkvOutput(last_hidden_state=hidden_states, state=state, hidden_states=all_hidden_states, attentions=all_self_attentions)

    def _rescale_layers(self):
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for block_id, block in enumerate(self.blocks):
                    if self.training:
                        block.attention.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    elif hasattr(block.attention.output.weight, 'SCB'):
                        block.attention.output.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                    elif hasattr(block.attention.output.weight, 'quant_state'):
                        self._bnb_4bit_dequantize_and_rescale(block.attention.output, block_id)
                        self._bnb_4bit_dequantize_and_rescale(block.feed_forward.value, block_id)
                    else:
                        block.attention.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.div_(2 ** int(block_id // self.config.rescale_every))
        self.layers_are_rescaled = not self.training

    def _bnb_4bit_dequantize_and_rescale(self, target_layer, block_id):
        """
        Perform the dequantization and rescaling of the weights of a given layer. After that operation the layer will
        be quantized again.
        """
        if not is_bitsandbytes_available():
            raise ImportError('Please install bitsandbytes to use this method.')
        import bitsandbytes as bnb
        dequant_weights = bnb.functional.dequantize_4bit(target_layer.weight.data, target_layer.weight.quant_state)
        dequant_weights.div_(2 ** int(block_id // self.config.rescale_every))
        quant_weight = bnb.nn.Params4bit(dequant_weights.to('cpu'), requires_grad=False).to(dequant_weights.device)
        setattr(target_layer, 'weight', quant_weight)