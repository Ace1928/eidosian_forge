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
@classmethod
def _object_compiled_hook(cls, ll_module, buf):
    """
        `ll_module` was compiled into object code `buf`.
        """
    try:
        self = ll_module.__library
    except AttributeError:
        return
    if self._object_caching_enabled:
        self._compiled = True
        self._compiled_object = buf