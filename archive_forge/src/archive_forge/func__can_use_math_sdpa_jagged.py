import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _can_use_math_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    if not params.query.transpose(1, 2).is_contiguous() or not params.key.transpose(1, 2).is_contiguous() or (not params.value.transpose(1, 2).is_contiguous()):
        if debug:
            log.warning('If inputs are nested tensors they must be contiguous after transposing.')
        return False
    if params.is_causal:
        if debug:
            log.warning('Nested tensors for query / key are not supported when is_causal=True.')
        return False
    return True