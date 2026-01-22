from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.core.imputils import lower_constant, lower_cast
from numba.core.ccallback import CFunc
from numba.core import cgutils
from llvmlite import ir
from numba.core import types
from numba.core.types import (FunctionType, UndefinedFunctionType,
from numba.core.dispatcher import Dispatcher
def _get_wrapper_address(func, sig):
    """Return the address of a compiled cfunc wrapper function of `func`.

    Warning: The compiled function must be compatible with the given
    signature `sig`. If it is not, then result of calling the compiled
    function is undefined. The compatibility is ensured when passing
    in a first-class function to a Numba njit compiled function either
    as an argument or via namespace scoping.

    Parameters
    ----------
    func : object
      A Numba cfunc or jit decoreated function or an object that
      implements the wrapper address protocol (see note below).
    sig : Signature
      The expected function signature.

    Returns
    -------
    addr : int
      An address in memory (pointer value) of the compiled function
      corresponding to the specified signature.

    Note: wrapper address protocol
    ------------------------------

    An object implements the wrapper address protocol iff the object
    provides a callable attribute named __wrapper_address__ that takes
    a Signature instance as the argument, and returns an integer
    representing the address or pointer value of a compiled function
    for the given signature.

    """
    if not sig.is_precise():
        addr = -1
    elif hasattr(func, '__wrapper_address__'):
        addr = func.__wrapper_address__()
    elif isinstance(func, CFunc):
        assert sig == func._sig
        addr = func.address
    elif isinstance(func, Dispatcher):
        cres = func.get_compile_result(sig)
        wrapper_name = cres.fndesc.llvm_cfunc_wrapper_name
        addr = cres.library.get_pointer_to_function(wrapper_name)
    else:
        raise NotImplementedError(f'get wrapper address of {type(func)} instance with {sig!r}')
    if not isinstance(addr, int):
        raise TypeError(f'wrapper address must be integer, got {type(addr)} instance')
    if addr <= 0 and addr != -1:
        raise ValueError(f'wrapper address of {type(func)} instance must be a positive integer but got {addr} [sig={sig}]')
    return addr