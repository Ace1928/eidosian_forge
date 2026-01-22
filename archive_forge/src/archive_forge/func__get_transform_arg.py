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
def _get_transform_arg(py_func):
    """Return function that transform arg into index"""
    args = inspect.getfullargspec(py_func).args
    pos_by_arg = {arg: i for i, arg in enumerate(args)}

    def transform_arg(arg):
        if isinstance(arg, int):
            return arg
        try:
            return pos_by_arg[arg]
        except KeyError:
            msg = f'Specified writable arg {arg} not found in arg list {args} for function {py_func.__qualname__}'
            raise RuntimeError(msg)
    return transform_arg