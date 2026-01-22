from functools import partial
from typing import List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
import torch._refs.linalg as linalg
from torch import Tensor
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
def _check_norm_dtype(dtype: Optional[torch.dtype], x_dtype: torch.dtype, fn_name: str):
    """
    Checks related to the dtype kwarg in `linalg.*norm` functions
    """
    if dtype is not None:
        torch._check(utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype), lambda: f'{fn_name}: dtype should be floating point or complex. Got {dtype}')
        torch._check(utils.is_complex_dtype(dtype) == utils.is_complex_dtype(x_dtype), lambda: '{fn_name}: dtype should be {d} for {d} inputs. Got {dtype}'.format(fn_name=fn_name, d='complex' if utils.is_complex_dtype(x_dtype) else 'real', dtype=dtype))
        torch._check(utils.get_higher_dtype(dtype, x_dtype) == dtype, lambda: f'{fn_name}: the dtype of the input ({x_dtype}) should be convertible without narrowing to the specified dtype ({{dtype}})')