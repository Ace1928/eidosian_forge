import logging
import os
import sys
from llvmlite import ir
from llvmlite.binding import Linkage
from numba.pycc import llvm_types as lt
from numba.core.cgutils import create_constant_array
from numba.core.compiler import compile_extra, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.registry import cpu_target
from numba.core.runtime import nrtdynmod
from numba.core import cgutils
def _emit_module_init_code(self, llvm_module, builder, modobj, method_array, env_array, envgv_array):
    """
        Emit call to "external" init function, if any.
        """
    if self.external_init_function:
        fnty = ir.FunctionType(lt._int32, [modobj.type, self.method_def_ptr, self.env_def_ptr, envgv_array.type])
        fn = ir.Function(llvm_module, fnty, self.external_init_function)
        return builder.call(fn, [modobj, method_array, env_array, envgv_array])
    else:
        return None