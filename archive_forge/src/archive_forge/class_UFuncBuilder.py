import inspect
import warnings
from contextlib import contextmanager
from numba.core import config, targetconfig
from numba.core.decorators import jit
from numba.core.descriptors import TargetDescriptor
from numba.core.extending import is_jitted
from numba.core.errors import NumbaDeprecationWarning
from numba.core.options import TargetOptions, include_default_options
from numba.core.registry import cpu_target
from numba.core.target_extension import dispatcher_registry, target_registry
from numba.core import utils, types, serialize, compiler, sigutils
from numba.np.numpy_support import as_dtype
from numba.np.ufunc import _internal
from numba.np.ufunc.sigparse import parse_signature
from numba.np.ufunc.wrappers import build_ufunc_wrapper, build_gufunc_wrapper
from numba.core.caching import FunctionCache, NullCache
from numba.core.compiler_lock import global_compiler_lock
class UFuncBuilder(_BaseUFuncBuilder):

    def __init__(self, py_func, identity=None, cache=False, targetoptions={}):
        if is_jitted(py_func):
            py_func = py_func.py_func
        self.py_func = py_func
        self.identity = parse_identity(identity)
        with _suppress_deprecation_warning_nopython_not_supplied():
            self.nb_func = jit(_target='npyufunc', cache=cache, **targetoptions)(py_func)
        self._sigs = []
        self._cres = {}

    def _finalize_signature(self, cres, args, return_type):
        """Slated for deprecation, use ufuncbuilder._finalize_ufunc_signature()
        instead.
        """
        return _finalize_ufunc_signature(cres, args, return_type)

    def build_ufunc(self):
        with global_compiler_lock:
            dtypelist = []
            ptrlist = []
            if not self.nb_func:
                raise TypeError('No definition')
            keepalive = []
            cres = None
            for sig in self._sigs:
                cres = self._cres[sig]
                dtypenums, ptr, env = self.build(cres, sig)
                dtypelist.append(dtypenums)
                ptrlist.append(int(ptr))
                keepalive.append((cres.library, env))
            datlist = [None] * len(ptrlist)
            if cres is None:
                argspec = inspect.getfullargspec(self.py_func)
                inct = len(argspec.args)
            else:
                inct = len(cres.signature.args)
            outct = 1
            ufunc = _internal.fromfunc(self.py_func.__name__, self.py_func.__doc__, ptrlist, dtypelist, inct, outct, datlist, keepalive, self.identity)
            return ufunc

    def build(self, cres, signature):
        """Slated for deprecation, use
        ufuncbuilder._build_element_wise_ufunc_wrapper().
        """
        return _build_element_wise_ufunc_wrapper(cres, signature)