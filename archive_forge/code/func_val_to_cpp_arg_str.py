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
def val_to_cpp_arg_str(self, type_, val, is_legacy_abi) -> str:
    if config.aot_inductor.abi_compatible and (not is_legacy_abi) and isinstance(type_, torch.OptionalType):
        if val is None:
            return '0'
        if isinstance(val, (bool, int, str, float)):
            var_name = f'var_{next(self.arg_var_id)}'
            self.writeline(f'auto {var_name} = {self.val_to_arg_str(val)};')
            return f'&{var_name}'
        if not isinstance(type_.getElementType(), torch.TensorType):
            return f'&{self.val_to_arg_str(val)}'
    return self.val_to_arg_str(val)