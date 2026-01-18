import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import (
from torch._prims_common import (
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
from .utils import (
from .virtualized import ops, V
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
def var_mean_welford_(x, axis, *, correction, keepdim, return_mean):
    if correction is None:
        correction = 1
    kwargs = _make_reduction_inner(x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None)
    loader = kwargs.pop('inner_fn')
    kwargs.pop('dst_dtype')
    kwargs.pop('src_dtype')
    mean, m2, _ = ir.WelfordReduction.create(inner_fns=(loader,), reduction_type='welford_reduce', dtype=x.get_dtype(), **kwargs)
    m2.realize()
    dtype = x.get_dtype()
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    rnumel = sympy_product((size[i] for i in axis))

    def get_constant_or_index_expr(x, dtype):
        if isinstance(x, sympy.Expr) and (not x.is_number):
            return ops.to_dtype(ops.index_expr(x, torch.int64), dtype)
        return ops.constant(x, dtype)

    def scale_fn(data):
        c = get_constant_or_index_expr(correction, dtype)
        N = get_constant_or_index_expr(rnumel, dtype)
        zero = ops.constant(0, dtype)
        return data / ops.maximum(zero, N - c)
    var = make_pointwise(scale_fn)(m2)
    if return_mean:
        mean.realize()
        return (var, mean)
    return var