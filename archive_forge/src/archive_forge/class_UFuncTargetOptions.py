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
class UFuncTargetOptions(_options_mixin, TargetOptions):

    def finalize(self, flags, options):
        if not flags.is_set('enable_pyobject'):
            flags.enable_pyobject = True
        if not flags.is_set('enable_looplift'):
            flags.enable_looplift = True
        flags.inherit_if_not_set('nrt', default=True)
        if not flags.is_set('debuginfo'):
            flags.debuginfo = config.DEBUGINFO_DEFAULT
        if not flags.is_set('boundscheck'):
            flags.boundscheck = flags.debuginfo
        flags.enable_pyobject_looplift = True
        flags.inherit_if_not_set('fastmath')