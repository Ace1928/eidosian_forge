from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from .attn_bias import AttentionBias
from .common import (
An operator optimized for very small values of K (``K <= 32``)         and f32 pre-Ampere as it does not use TensorCores.
    Only supports contiguous inputs in BMK format, so an extra reshape         or contiguous call might be done.

    :Deprecated:

        This operator is deprecated and should not be used in new code
    