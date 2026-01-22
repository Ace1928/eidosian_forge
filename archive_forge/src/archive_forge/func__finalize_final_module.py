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
def _finalize_final_module(self):
    """
        Make the underlying LLVM module ready to use.
        """
    self._finalize_dynamic_globals()
    self._verify_declare_only_symbols()
    self._final_module.__library = weakref.proxy(self)
    cleanup = self._codegen._add_module(self._final_module)
    if cleanup:
        weakref.finalize(self, cleanup)
    self._finalize_specific()
    self._finalized = True
    if config.DUMP_OPTIMIZED:
        dump('OPTIMIZED DUMP %s' % self.name, self.get_llvm_str(), 'llvm')
    if config.DUMP_ASSEMBLY:
        dump('ASSEMBLY %s' % self.name, self.get_asm_str(), 'asm')