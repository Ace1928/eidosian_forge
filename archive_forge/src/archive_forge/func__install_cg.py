import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def _install_cg(self, targetctx=None):
    """
        Install an implementation function for a DUFunc object in the
        given target context.  If no target context is given, then
        _install_cg() installs into the target context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
    if targetctx is None:
        targetctx = self._dispatcher.targetdescr.target_context
    _any = types.Any
    _arr = types.Array
    sig0 = (_any,) * self.ufunc.nin + (_arr,) * self.ufunc.nout
    sig1 = (_any,) * self.ufunc.nin
    targetctx.insert_func_defn([(self._lower_me, self, sig) for sig in (sig0, sig1)])