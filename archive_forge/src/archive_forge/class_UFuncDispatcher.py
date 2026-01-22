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
class UFuncDispatcher(serialize.ReduceMixin):
    """
    An object handling compilation of various signatures for a ufunc.
    """
    targetdescr = ufunc_target

    def __init__(self, py_func, locals={}, targetoptions={}):
        self.py_func = py_func
        self.overloads = utils.UniqueDict()
        self.targetoptions = targetoptions
        self.locals = locals
        self.cache = NullCache()

    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
        return dict(pyfunc=self.py_func, locals=self.locals, targetoptions=self.targetoptions)

    @classmethod
    def _rebuild(cls, pyfunc, locals, targetoptions):
        """
        NOTE: part of ReduceMixin protocol
        """
        return cls(py_func=pyfunc, locals=locals, targetoptions=targetoptions)

    def enable_caching(self):
        self.cache = FunctionCache(self.py_func)

    def compile(self, sig, locals={}, **targetoptions):
        locs = self.locals.copy()
        locs.update(locals)
        topt = self.targetoptions.copy()
        topt.update(targetoptions)
        flags = compiler.Flags()
        self.targetdescr.options.parse_as_flags(flags, topt)
        flags.no_cpython_wrapper = True
        flags.error_model = 'numpy'
        flags.enable_looplift = False
        return self._compile_core(sig, flags, locals)

    def _compile_core(self, sig, flags, locals):
        """
        Trigger the compiler on the core function or load a previously
        compiled version from the cache.  Returns the CompileResult.
        """
        typingctx = self.targetdescr.typing_context
        targetctx = self.targetdescr.target_context

        @contextmanager
        def store_overloads_on_success():
            try:
                yield
            except Exception:
                raise
            else:
                exists = self.overloads.get(cres.signature)
                if exists is None:
                    self.overloads[cres.signature] = cres
        with global_compiler_lock:
            with targetconfig.ConfigStack().enter(flags.copy()):
                with store_overloads_on_success():
                    cres = self.cache.load_overload(sig, targetctx)
                    if cres is not None:
                        return cres
                    args, return_type = sigutils.normalize_signature(sig)
                    cres = compiler.compile_extra(typingctx, targetctx, self.py_func, args=args, return_type=return_type, flags=flags, locals=locals)
                    self.cache.save_overload(sig, cres)
                    return cres