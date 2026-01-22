import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def alloca_once(builder, ty, size=None, name='', zfill=False):
    """Allocate stack memory at the entry block of the current function
    pointed by ``builder`` with llvm type ``ty``.  The optional ``size`` arg
    set the number of element to allocate.  The default is 1.  The optional
    ``name`` arg set the symbol name inside the llvm IR for debugging.
    If ``zfill`` is set, fill the memory with zeros at the current
    use-site location.  Note that the memory is always zero-filled after the
    ``alloca`` at init-site (the entry block).
    """
    if isinstance(size, int):
        size = ir.Constant(intp_t, size)
    with debuginfo.suspend_emission(builder):
        with builder.goto_entry_block():
            ptr = builder.alloca(ty, size=size, name=name)
            builder.store(ty(None), ptr)
        if zfill:
            builder.store(ptr.type.pointee(None), ptr)
        return ptr