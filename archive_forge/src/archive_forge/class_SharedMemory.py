from typing import Any, Mapping
import warnings
import cupy
from cupy_backends.cuda.api import runtime
from cupy.cuda import device
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit._internal_types import BuiltinFunc
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit._internal_types import Range
from cupyx.jit import _compile
from functools import reduce
class SharedMemory(BuiltinFunc):

    def __call__(self, dtype, size, alignment=None):
        """Allocates shared memory and returns it as a 1-D array.

        Args:
            dtype (dtype):
                The dtype of the returned array.
            size (int or None):
                If ``int`` type, the size of static shared memory.
                If ``None``, declares the shared memory with extern specifier.
            alignment (int or None): Enforce the alignment via __align__(N).
        """
        super().__call__()

    def call_const(self, env, dtype, size, alignment=None):
        name = env.get_fresh_variable_name(prefix='_smem')
        ctype = _cuda_typerules.to_ctype(dtype)
        var = Data(name, _cuda_types.SharedMem(ctype, size, alignment))
        env.decls[name] = var
        env.locals[name] = var
        return Data(name, _cuda_types.Ptr(ctype))