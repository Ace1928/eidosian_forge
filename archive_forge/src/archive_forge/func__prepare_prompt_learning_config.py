import copy
import inspect
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple
import accelerate
import torch
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import storage_ptr, storage_size
from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
from .constants import (
def _prepare_prompt_learning_config(peft_config, model_config):
    if peft_config.num_layers is None:
        if 'num_hidden_layers' in model_config:
            num_layers = model_config['num_hidden_layers']
        elif 'num_layers' in model_config:
            num_layers = model_config['num_layers']
        elif 'n_layer' in model_config:
            num_layers = model_config['n_layer']
        else:
            raise ValueError('Please specify `num_layers` in `peft_config`')
        peft_config.num_layers = num_layers
    if peft_config.token_dim is None:
        if 'hidden_size' in model_config:
            token_dim = model_config['hidden_size']
        elif 'n_embd' in model_config:
            token_dim = model_config['n_embd']
        elif 'd_model' in model_config:
            token_dim = model_config['d_model']
        else:
            raise ValueError('Please specify `token_dim` in `peft_config`')
        peft_config.token_dim = token_dim
    if peft_config.num_attention_heads is None:
        if 'num_attention_heads' in model_config:
            num_attention_heads = model_config['num_attention_heads']
        elif 'n_head' in model_config:
            num_attention_heads = model_config['n_head']
        elif 'num_heads' in model_config:
            num_attention_heads = model_config['num_heads']
        elif 'encoder_attention_heads' in model_config:
            num_attention_heads = model_config['encoder_attention_heads']
        else:
            raise ValueError('Please specify `num_attention_heads` in `peft_config`')
        peft_config.num_attention_heads = num_attention_heads
    if getattr(peft_config, 'encoder_hidden_size', None) is None:
        setattr(peft_config, 'encoder_hidden_size', peft_config.token_dim)
    return peft_config