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
@register_lowering(aten._adaptive_avg_pool2d)
def _adaptive_avg_pool2d(x, output_size):
    assert isinstance(x, TensorBox)
    assert len(output_size) == 2
    x.realize_hint()
    *batch, h_in, w_in = x.get_size()
    h_in = V.graph.sizevars.evaluate_static_shape(h_in)
    w_in = V.graph.sizevars.evaluate_static_shape(w_in)
    h_out, w_out = output_size
    if h_in == h_out and w_in == w_out:
        return clone(x)
    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return empty(o_size, dtype=x.get_dtype(), device=x.get_device())
    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [h_in // h_out, w_in // w_out]
        return avg_pool2d(x, kernel_size)
    h_kernel_max = ceildiv(h_in + h_out - 1, h_out)
    w_kernel_max = ceildiv(w_in + w_out - 1, w_out)
    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    def start_index(index, out_dim, inp_dim):
        return FloorDiv(index * inp_dim, out_dim)

    def end_index(index, out_dim, inp_dim):
        return FloorDiv((index + 1) * inp_dim + out_dim - 1, out_dim)
    h_start_index = functools.partial(start_index, out_dim=h_out, inp_dim=h_in)
    h_end_index = functools.partial(end_index, out_dim=h_out, inp_dim=h_in)
    w_start_index = functools.partial(start_index, out_dim=w_out, inp_dim=w_in)
    w_end_index = functools.partial(end_index, out_dim=w_out, inp_dim=w_in)
    window_size = h_kernel_max * w_kernel_max
    if window_size > 25:
        return fallback_adaptive_avg_pool2d(x, output_size)
    fn_sum = _adaptive_pooling_idx_sum([h_kernel_max, w_kernel_max], [h_start_index, w_start_index], [h_end_index, w_end_index])
    ones_loader = pad_adaptive_loader(ones_like(x))

    def fn(idx):
        return ops.truediv(fn_sum(idx, pad_adaptive_loader(x)), fn_sum(idx, ones_loader))
    rv = Pointwise.create(device=x.get_device(), dtype=dtype, inner_fn=fn, ranges=new_size)
    return rv