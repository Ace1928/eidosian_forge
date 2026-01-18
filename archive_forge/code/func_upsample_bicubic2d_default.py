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
@register_lowering(aten.upsample_bicubic2d.default)
def upsample_bicubic2d_default(x, output_size, align_corners: bool, scales_h: Optional[float]=None, scales_w: Optional[float]=None):
    x.realize_hint()
    x_loader = x.make_loader()
    N, C, iH, iW = x.get_size()
    oH, oW = output_size
    iH = V.graph.sizevars.evaluate_static_shape(iH)
    iW = V.graph.sizevars.evaluate_static_shape(iW)

    def get_int_dtype(maxval):
        if maxval > torch.iinfo(torch.int32).max:
            return torch.int64
        return torch.int32

    def compute_scale(in_size, out_size, align_corners, scale=None):
        if align_corners:
            return (in_size - 1) / (out_size - 1) if out_size > 1 else 0
        else:
            return 1 / scale if scale is not None and scale > 0 else in_size / out_size

    def compute_source_index(scale, dst_index, align_corners):
        dst_index_ie = ops.index_expr(dst_index, torch.float32)
        scale = ops.constant(scale, torch.float32)
        if align_corners:
            return ops.mul(scale, dst_index_ie)
        else:
            half = ops.constant(0.5, torch.float32)
            return scale * (dst_index_ie + half) - half

    def cubic_convolution1(x, A):
        _Ap2, _Ap3, _1 = _create_constants(A + 2, A + 3, 1, dtype=torch.float32)
        return (_Ap2 * x - _Ap3) * x * x + _1

    def cubic_convolution2(x, A):
        _A, _4A, _5A, _8A = _create_constants(A, 4 * A, 5 * A, 8 * A, dtype=torch.float32)
        return ((_A * x - _5A) * x + _8A) * x - _4A

    def get_cubic_upsample_coefficients(t):
        A = -0.75
        _1 = ops.constant(1.0, torch.float32)
        c0 = cubic_convolution2(ops.add(t, _1), A)
        c1 = cubic_convolution1(t, A)
        x2 = ops.sub(_1, t)
        c2 = cubic_convolution1(x2, A)
        c3 = cubic_convolution2(ops.add(x2, _1), A)
        return (c0, c1, c2, c3)

    def cubic_interp1d(xs, t):
        cs = get_cubic_upsample_coefficients(t)
        return xs[0] * cs[0] + xs[1] * cs[1] + xs[2] * cs[2] + xs[3] * cs[3]
    height_scale = compute_scale(iH, oH, align_corners, scales_h)
    width_scale = compute_scale(iW, oW, align_corners, scales_h)

    def clamp(v, min, max):
        return ops.maximum(min, ops.minimum(max, v))

    def fn(idx):
        n, c, oy, ox = idx
        real_x = compute_source_index(width_scale, ox, align_corners)
        in_x = ops.floor(real_x)
        t_x = ops.sub(real_x, in_x)
        real_y = compute_source_index(height_scale, oy, align_corners)
        in_y = ops.floor(real_y)
        t_y = ops.sub(real_y, in_y)

        def load_bounded(fy, fx):
            _0 = ops.constant(0, torch.int32)
            iHm1 = ops.constant(iH - 1, torch.int32)
            iWm1 = ops.constant(iW - 1, torch.int32)
            iy = ops.indirect_indexing(clamp(fy, _0, iHm1), iH, check=False)
            ix = ops.indirect_indexing(clamp(fx, _0, iWm1), iW, check=False)
            return x_loader([n, c, iy, ix])
        iy = ops.to_dtype(in_y, get_int_dtype(iH + 1))
        ix = ops.to_dtype(in_x, get_int_dtype(iW + 1))
        iys_ofs = tuple((ops.add(iy, ofs) for ofs in (-1, 0, 1, 2)))
        ixs_ofs = tuple((ops.add(ix, ofs) for ofs in (-1, 0, 1, 2)))

        def get_x_interp(y):
            coeffs_x = tuple((load_bounded(y, x) for x in ixs_ofs))
            return cubic_interp1d(coeffs_x, t_x)
        coeffs_y = tuple((get_x_interp(y) for y in iys_ofs))
        return cubic_interp1d(coeffs_y, t_y)
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=[N, C, sympy.Integer(oH), sympy.Integer(oW)])