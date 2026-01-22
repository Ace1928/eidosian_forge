import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
def _reduction_int_to_str(reduction: int) -> str:
    from torch._decomp.decompositions import Reduction
    if reduction == Reduction.NONE.value:
        return 'none'
    elif reduction == Reduction.MEAN.value:
        return 'mean'
    elif reduction == Reduction.SUM.value:
        return 'sum'
    else:
        raise ValueError(f'{reduction} is not a valid value for reduction')