import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta
def val_to_arg_str(self, val) -> str:
    if val is None:
        if config.aot_inductor.abi_compatible:
            return '0'
        return 'c10::nullopt'
    elif isinstance(val, bool):
        if config.aot_inductor.abi_compatible:
            return '1' if val else '0'
        else:
            return 'true' if val else 'false'
    elif isinstance(val, int):
        return f'{val}L'
    elif isinstance(val, str):
        return f'"{val}"'
    elif isinstance(val, (ComputedBuffer, InputBuffer, ReinterpretView)):
        return val.codegen_reference()
    elif isinstance(val, torch.device):
        return self.codegen_device(val)
    elif isinstance(val, torch.dtype):
        return self.codegen_dtype(val)
    elif isinstance(val, float) and val in [float('inf'), float('-inf')]:
        if val == float('inf'):
            return 'std::numeric_limits<float>::infinity()'
        else:
            return '-std::numeric_limits<float>::infinity()'
    elif isinstance(val, (list, tuple)):
        result = f'{{{', '.join((self.val_to_arg_str(x) for x in val))}}}'
        if config.aot_inductor.abi_compatible:
            return f'{self.codegen_int_array_var(result)}, {len(val)}'
        else:
            return result
    else:
        return repr(val)