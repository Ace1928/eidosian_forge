import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def type_casts(f: Callable, type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND, compute_dtype_only: bool=False):

    @functools.wraps(f)
    def inner(*args, **kwargs):
        flat_args = [x for x in pytree.arg_tree_leaves(*args, **kwargs) if isinstance(x, Tensor)]
        computation_dtype, result_dtype = utils.elementwise_dtypes(*flat_args, type_promotion_kind=type_promotion)

        def increase_prec(x):
            if isinstance(x, Tensor):
                return x.to(computation_dtype)
            else:
                return x

        def decrease_prec(x):
            if isinstance(x, Tensor):
                return x.to(result_dtype)
            else:
                return x
        r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
        if compute_dtype_only:
            return r
        else:
            return tree_map(decrease_prec, r)
    return inner