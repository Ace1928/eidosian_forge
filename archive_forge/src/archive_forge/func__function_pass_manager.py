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
def _function_pass_manager(self, llvm_module, **kwargs):
    pm = ll.create_function_pass_manager(llvm_module)
    pm.add_target_library_info(llvm_module.triple)
    self._tm.add_analysis_passes(pm)
    with self._pass_manager_builder(**kwargs) as pmb:
        pmb.populate(pm)
    if config.LLVM_REFPRUNE_PASS:
        pm.add_refprune_pass(_parse_refprune_flags())
    return pm