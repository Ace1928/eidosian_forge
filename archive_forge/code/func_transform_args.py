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
def transform_args(args, broadcast, type_promotion_kind, convert_input_to_bool):
    indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    if (type_promotion_kind or convert_input_to_bool) and indices:
        if convert_input_to_bool:
            dtype = torch.bool
        else:
            promoting_args = [a for a in args if isinstance(a, Number) or hasattr(a, 'get_dtype')]
            dtype = get_promoted_dtype(*promoting_args, type_promotion_kind=type_promotion_kind)

        def promote(arg):
            if isinstance(arg, TensorBox):
                return to_dtype(arg, dtype)
            elif isinstance(arg, ir.Constant):
                return ir.Constant(arg.value, dtype, args[indices[0]].get_device())
            else:
                return arg
        args = [promote(a) for a in args]
    if broadcast and indices:
        for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
            args[i] = x
        for i in range(len(args)):
            if isinstance(args[i], ir.Constant):
                args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
    return args