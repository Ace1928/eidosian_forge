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
@contextmanager
def _suppress_deprecation_warning_nopython_not_supplied():
    """This suppresses the NumbaDeprecationWarning that occurs through the use
    of `jit` without the `nopython` kwarg. This use of `jit` occurs in a few
    places in the `{g,}ufunc` mechanism in Numba, predominantly to wrap the
    "kernel" function."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=NumbaDeprecationWarning, message=".*The 'nopython' keyword argument was not supplied*")
        yield