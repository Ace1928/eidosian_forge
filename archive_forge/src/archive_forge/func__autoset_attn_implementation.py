import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zipfile import is_zipfile
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint
from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .pytorch_utils import (  # noqa: F401
from .quantizers import AutoHfQuantizer, HfQuantizer
from .safetensors_conversion import auto_conversion
from .utils import (
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
@classmethod
def _autoset_attn_implementation(cls, config, use_flash_attention_2: bool=False, torch_dtype: Optional[torch.dtype]=None, device_map: Optional[Union[str, Dict[str, int]]]=None, check_device_map: bool=True):
    """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
    requested_attn_implementation = None
    if hasattr(config, '_attn_implementation_internal') and config._attn_implementation_internal is not None:
        if config._attn_implementation != 'flash_attention_2' and use_flash_attention_2:
            raise ValueError(f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible. We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.')
        if config._attn_implementation not in ['eager', 'sdpa', 'flash_attention_2']:
            message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation)'
            if cls._supports_flash_attn_2:
                message += ', `"attn_implementation=flash_attention_2"` (implementation using flash attention 2)'
            if cls._supports_sdpa:
                message += ', `"attn_implementation=sdpa"` (implementation using torch.nn.functional.scaled_dot_product_attention)'
            raise ValueError(message + '.')
        requested_attn_implementation = config._attn_implementation_internal
    if use_flash_attention_2:
        logger.warning_once('The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.')
        config._attn_implementation = 'flash_attention_2'
    if config._attn_implementation == 'flash_attention_2':
        cls._check_and_enable_flash_attn_2(config, torch_dtype=torch_dtype, device_map=device_map, hard_check_only=False, check_device_map=check_device_map)
    elif requested_attn_implementation in [None, 'sdpa']:
        config = cls._check_and_enable_sdpa(config, hard_check_only=False if requested_attn_implementation is None else True)
    else:
        config._attn_implementation = 'eager'
    return config