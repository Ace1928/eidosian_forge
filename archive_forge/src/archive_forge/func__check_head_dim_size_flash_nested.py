import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def _check_head_dim_size_flash_nested(params: SDPAParams, debug=False) -> bool:
    max_size = 256
    query_size_last = params.query.size(-1)
    key_size_last = params.key.size(-1)
    value_size_last = params.value.size(-1)
    same_head_dim_size = query_size_last == key_size_last and query_size_last == value_size_last
    if not (same_head_dim_size and query_size_last % 8 == 0 and (query_size_last <= max_size)):
        if debug:
            log.warning('For NestedTensor inputs, Flash attention requires q,k,v to have the same last dimension and to be a multiple of 8 and less than or equal to 256. Got Query.size(-1): %d, Key.size(-1): %d, Value.size(-1): %d instead.', query_size_last, key_size_last, value_size_last)
        return False
    return True