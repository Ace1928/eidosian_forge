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
class JitEngine(object):
    """Wraps an ExecutionEngine to provide custom symbol tracking.
    Since the symbol tracking is incomplete  (doesn't consider
    loaded code object), we are not putting it in llvmlite.
    """

    def __init__(self, ee):
        self._ee = ee
        self._defined_symbols = set()

    def is_symbol_defined(self, name):
        """Is the symbol defined in this session?
        """
        return name in self._defined_symbols

    def _load_defined_symbols(self, mod):
        """Extract symbols from the module
        """
        for gsets in (mod.functions, mod.global_variables):
            self._defined_symbols |= {gv.name for gv in gsets if not gv.is_declaration}

    def add_module(self, module):
        """Override ExecutionEngine.add_module
        to keep info about defined symbols.
        """
        self._load_defined_symbols(module)
        return self._ee.add_module(module)

    def add_global_mapping(self, gv, addr):
        """Override ExecutionEngine.add_global_mapping
        to keep info about defined symbols.
        """
        self._defined_symbols.add(gv.name)
        return self._ee.add_global_mapping(gv, addr)
    set_object_cache = _proxy(ll.ExecutionEngine.set_object_cache)
    finalize_object = _proxy(ll.ExecutionEngine.finalize_object)
    get_function_address = _proxy(ll.ExecutionEngine.get_function_address)
    get_global_value_address = _proxy(ll.ExecutionEngine.get_global_value_address)