import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _check_batch_size_nested(params: SDPAParams, debug=False) -> bool:
    q_batch_size = params.query.size(0)
    k_batch_size = params.key.size(0)
    v_batch_size = params.value.size(0)
    return q_batch_size == k_batch_size and q_batch_size == v_batch_size