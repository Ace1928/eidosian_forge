import ctypes
from functools import cached_property
from numba.core import compiler, registry
from numba.core.caching import NullCache, FunctionCache
from numba.core.dispatcher import _FunctionCompiler
from numba.core.typing import signature
from numba.core.typing.ctypes_utils import to_ctypes
from numba.core.compiler_lock import global_compiler_lock
class _CFuncCompiler(_FunctionCompiler):

    def _customize_flags(self, flags):
        flags.no_cpython_wrapper = True
        flags.no_cfunc_wrapper = False
        flags.no_compile = True
        flags.enable_pyobject = False
        if flags.force_pyobject:
            raise NotImplementedError('object mode not allowed in C callbacks')
        return flags