import math
import functools
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
from typing_extensions import ParamSpec, Self, TypeAlias
import torch
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch.utils._foreach_utils import (
from torch._utils import is_compiling
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
@staticmethod
def _process_value_according_to_param_policy(param: torch.Tensor, value: torch.Tensor, param_id: int, param_groups: List[Dict[Any, Any]], key: Hashable=None) -> torch.Tensor:
    fused = False
    capturable = False
    assert param_groups is not None
    for pg in param_groups:
        if param_id in pg['params']:
            fused = pg['fused'] if 'fused' in pg else False
            capturable = pg['capturable'] if 'capturable' in pg else False
            break
    if key == 'step':
        if capturable or fused:
            return value.to(dtype=torch.float32, device=param.device)
        else:
            return value
    elif param.is_floating_point():
        return value.to(dtype=param.dtype, device=param.device)
    else:
        return value.to(device=param.device)