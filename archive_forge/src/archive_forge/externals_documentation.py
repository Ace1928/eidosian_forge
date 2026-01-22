import sys
from llvmlite import ir
import llvmlite.binding as ll
from numba.core import utils, intrinsics
from numba import _helperlib

        Install the functions into LLVM.  This only needs to be done once,
        as the mappings are persistent during the process lifetime.
        