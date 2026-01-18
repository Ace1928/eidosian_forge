from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding

    Compile all LLVM NRT functions and return a library containing them.
    The library is created using the given target context.
    