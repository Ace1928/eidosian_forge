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
def _parse_refprune_flags():
    """Parse refprune flags from the `config`.

    Invalid values are ignored an warn via a `NumbaInvalidConfigWarning`
    category.

    Returns
    -------
    flags : llvmlite.binding.RefPruneSubpasses
    """
    flags = config.LLVM_REFPRUNE_FLAGS.split(',')
    if not flags:
        return 0
    val = 0
    for item in flags:
        item = item.strip()
        try:
            val |= getattr(ll.RefPruneSubpasses, item.upper())
        except AttributeError:
            warnings.warn(f'invalid refprune flags {item!r}', NumbaInvalidConfigWarning)
    return val