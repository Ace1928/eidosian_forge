import warnings
import functools
import locale
import weakref
import ctypes
import html
import textwrap
import llvmlite.binding as ll
import llvmlite.ir as llvmir
from abc import abstractmethod, ABCMeta
from numba.core import utils, config, cgutils
from numba.core.llvm_bindings import create_pass_manager_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.core.runtime import rtsys
from numba.core.compiler_lock import require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning
from numba.misc.inspection import disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection
def _optimize_functions(self, ll_module):
    """
        Internal: run function-level optimizations inside *ll_module*.
        """
    ll_module.data_layout = self._codegen._data_layout
    with self._codegen._function_pass_manager(ll_module) as fpm:
        for func in ll_module.functions:
            k = f'Function passes on {func.name!r}'
            with self._recorded_timings.record(k):
                fpm.initialize()
                fpm.run(func)
                fpm.finalize()