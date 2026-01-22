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
def _get_resized_lm_head(self, old_lm_head: nn.Linear, new_num_tokens: Optional[int]=None, transposed: Optional[bool]=False) -> nn.Linear:
    """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
    if new_num_tokens is None:
        return old_lm_head
    if is_deepspeed_zero3_enabled():
        import deepspeed
        with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
            old_num_tokens, old_lm_head_dim = old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
    else:
        old_num_tokens, old_lm_head_dim = old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
    if old_num_tokens == new_num_tokens and (not is_deepspeed_zero3_enabled()):
        return old_lm_head
    if not isinstance(old_lm_head, nn.Linear):
        raise TypeError(f'Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You should either use a different resize function or make sure that `old_lm_head` are an instance of {nn.Linear}.')
    new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
    has_new_lm_head_bias = old_lm_head.bias is not None
    new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias, device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
    self._init_weights(new_lm_head)
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    if is_deepspeed_zero3_enabled():
        import deepspeed
        params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            self._copy_lm_head_original_to_resized(new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias)
    else:
        self._copy_lm_head_original_to_resized(new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias)
    return new_lm_head