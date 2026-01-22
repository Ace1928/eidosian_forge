from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def _compile_wrapper(self, wrapper_name):
    if self.is_parfors:
        wrapperlib = self.context.codegen().create_library(str(self))
        self._build_wrapper(wrapperlib, wrapper_name)
    else:
        wrapperlib = self.cache.load_overload(self.cres.signature, self.cres.target_context)
        if wrapperlib is None:
            wrapperlib = self.context.codegen().create_library(str(self))
            wrapperlib.enable_object_caching()
            self._build_wrapper(wrapperlib, wrapper_name)
            self.cache.save_overload(self.cres.signature, wrapperlib)
    return wrapperlib