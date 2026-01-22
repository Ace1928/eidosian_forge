import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config
from flash_attn.models.bigcode import remap_state_dict_hf_bigcode
from flash_attn.models.falcon import remap_state_dict_hf_falcon
from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
from flash_attn.models.gptj import remap_state_dict_hf_gptj
from flash_attn.models.llama import remap_state_dict_hf_llama
from flash_attn.models.opt import remap_state_dict_hf_opt
from flash_attn.modules.block import Block, ParallelBlock
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import (
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.utils.distributed import (
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.pretrained import state_dict_from_pretrained
class GPTPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError('Parameter config in `{}(config)` should be an instance of class `GPT2Config`. To create a model from a Google pretrained model use `model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(self.__class__.__name__, self.__class__.__name__))
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *args, strict=True, device=None, dtype=None, world_size=1, rank=0, **kwargs):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)
        state_dict = state_dict_from_pretrained(model_name, device='cpu', dtype=dtype)
        if model_name.startswith('gpt2'):
            state_dict = remap_state_dict_hf_gpt2(state_dict, config)
        elif model_name.startswith('facebook/opt'):
            state_dict = remap_state_dict_hf_opt(state_dict, config)
        elif model_name.startswith('EleutherAI/gpt-j-') or model_name.startswith('togethercomputer/GPT-JT-'):
            state_dict = remap_state_dict_hf_gptj(state_dict, config)
        elif model_name.startswith('EleutherAI/gpt-neox-') or model_name.startswith('EleutherAI/pythia-') or model_name.startswith('togethercomputer/RedPajama-INCITE-'):
            state_dict = remap_state_dict_hf_gpt_neox(state_dict, config)
        elif model_name.startswith('tiiuae/falcon-'):
            state_dict = remap_state_dict_hf_falcon(state_dict, config)
        elif model_name.startswith('meta-llama/Llama-'):
            state_dict = remap_state_dict_hf_llama(state_dict, config)
        elif model_name.startswith('bigcode/') or model_name.startswith('WizardLM/'):
            state_dict = remap_state_dict_hf_bigcode(state_dict, config)
        else:
            raise NotImplementedError(f'Model {model_name} not supported')
        if world_size > 1:
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model