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
def _check_llvm_bugs(self):
    """
        Guard against some well-known LLVM bug(s).
        """
    ir = '\n            define double @func()\n            {\n                ret double 1.23e+01\n            }\n            '
    mod = ll.parse_assembly(ir)
    ir_out = str(mod)
    if '12.3' in ir_out or '1.23' in ir_out:
        return
    if '1.0' in ir_out:
        loc = locale.getlocale()
        raise RuntimeError('LLVM will produce incorrect floating-point code in the current locale %s.\nPlease read https://numba.readthedocs.io/en/stable/user/faq.html#llvm-locale-bug for more information.' % (loc,))
    raise AssertionError('Unexpected IR:\n%s\n' % (ir_out,))