from typing import Any, List, Tuple
import torch.nn as nn
from torch.distributed.tensor.parallel._data_parallel_utils import (
def _update_module_param(param_list: List[Tuple[nn.Module, str, nn.Parameter]]):
    """
    Update parameters within the module
    """
    for item in param_list:
        parent_module, module_path, t = item
        assert hasattr(parent_module, module_path)
        delattr(parent_module, module_path)
        setattr(parent_module, module_path, t)