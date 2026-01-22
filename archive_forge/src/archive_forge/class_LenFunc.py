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
class LenFunc(BuiltinFunc):

    def call(self, env, *args, **kwds):
        if len(args) != 1:
            raise TypeError(f'len() expects only 1 argument, got {len(args)}')
        if kwds:
            raise TypeError('keyword arguments are not supported')
        arg = args[0]
        if not isinstance(arg.ctype, _cuda_types.CArray):
            raise TypeError('len() supports only array type')
        if not arg.ctype.ndim:
            raise TypeError('len() of unsized array')
        return Data(f'static_cast<long long>({arg.code}.shape()[0])', _cuda_types.Scalar('q'))