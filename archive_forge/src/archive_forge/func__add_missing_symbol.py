import sys
from llvmlite import ir
import llvmlite.binding as ll
from numba.core import utils, intrinsics
from numba import _helperlib
def _add_missing_symbol(symbol, addr):
    """Add missing symbol into LLVM internal symtab
    """
    if not ll.address_of_symbol(symbol):
        ll.add_symbol(symbol, addr)