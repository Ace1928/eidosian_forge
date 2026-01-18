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
def write_llvm_bitcode(self, output, wrap=False, **kws):
    self.export_python_wrap = wrap
    library = self._cull_exports()
    with open(output, 'wb') as fout:
        fout.write(library.emit_bitcode())