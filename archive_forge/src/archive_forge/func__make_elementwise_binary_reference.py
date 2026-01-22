import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def _make_elementwise_binary_reference(type_promotion_kind, aten_op=infer_aten_op, name=None, has_out=True, supports_lhs_python_scalar=True, supports_rhs_python_scalar=True, supports_two_python_scalars=False, should_register_decomposition=True) -> Callable:

    def inner(prim: Callable):
        nonlocal aten_op, name
        if name is None:
            name = prim.__name__

        @wraps(prim)
        @elementwise_type_promotion_wrapper(type_promoting_args=('a', 'b'), type_promotion_kind=type_promotion_kind)
        def _ref(a: Union[Tensor, NumberType], b: Union[Tensor, NumberType]) -> Tensor:
            torch._check_value(supports_lhs_python_scalar or not isinstance(a, Number), lambda: f'{name}: Received a lhs Python scalar to an elementwise binary operation that does not accept lhs scalars!')
            torch._check_value(supports_rhs_python_scalar or not isinstance(b, Number), lambda: f'{name}: Received a rhs Python scalar to an elementwise binary operation that does not accept rhs scalars!')
            torch._check_value(supports_two_python_scalars or not (isinstance(a, Number) and isinstance(b, Number)), lambda: f'{name}: Receive two Number inputs to an elementwise binary operation!')
            a, b = _maybe_broadcast(a, b)
            output = prim(a, b)
            return handle_noncontiguous_outputs([a, b], output)
        if has_out:
            _ref = out_wrapper()(_ref)
        _ref.__name__ = name
        if aten_op is infer_aten_op:
            aten_op = utils.get_aten_op(prim, name)
        if aten_op is not None and should_register_decomposition:
            register_decomposition(aten_op)(_ref)
        return _ref
    return inner