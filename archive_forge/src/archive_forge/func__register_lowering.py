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
def _register_lowering(aten_fn, decomp_fn, broadcast, type_promotion_kind, convert_input_to_bool):
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
        convert_input_to_bool: some logical ops require inputs are converted to bool
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        args: Union[List[Any], Tuple[Any, ...], Dict[Any, Any]] = list(args)
        unpacked = False
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = args[0]
        assert not any((x == 'out' for x in kwargs.keys())), "out= ops aren't yet supported"
        assert not any((isinstance(x, TensorBox) for x in kwargs.values())) or all((fn in fallbacks for fn in aten_fn))
        args = transform_args(args, broadcast, type_promotion_kind, convert_input_to_bool)
        if unpacked:
            args = [args]
        out = decomp_fn(*args, **kwargs)
        validate_ir(out)
        return out
    aten_fn = get_overloads(aten_fn)
    lowerings.update({fn: wrapped for fn in aten_fn})
    return wrapped