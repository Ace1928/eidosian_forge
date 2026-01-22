from typing import Any, List, Tuple
import torch.nn as nn
from torch.distributed.tensor.parallel._data_parallel_utils import (
def _localize_dtensor(module: nn.Module, *_: Any):
    """
    Convert DTensor parameters to local tensors
    """
    param_list = []
    for name, param in module.named_parameters():
        t, sharding_info = _flatten_tensor(param)
        if sharding_info is not None:
            t = nn.Parameter(t)
            t._st_info = sharding_info
            param_list.append((*_get_submodule_n_params(module, name), t))
    _update_module_param(param_list)