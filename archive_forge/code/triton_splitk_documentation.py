import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
Returns a triton.runtime.autotuner.AutoTuner.cache object, which
            represents mappings from kernel autotune keys (tuples describing kernel inputs)
            to triton.Config
            