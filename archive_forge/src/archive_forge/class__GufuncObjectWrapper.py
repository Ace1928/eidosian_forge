from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
class _GufuncObjectWrapper(_GufuncWrapper):

    def gen_loop_body(self, builder, pyapi, func, args):
        innercall, error = _prepare_call_to_object_mode(self.context, builder, pyapi, func, self.signature, args)
        return (innercall, error)

    def gen_prologue(self, builder, pyapi):
        self.gil = pyapi.gil_ensure()

    def gen_epilogue(self, builder, pyapi):
        pyapi.gil_release(self.gil)