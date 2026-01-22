import platform
import re
import sys
import warnings
from ._version import get_versions
from numba.misc.init_utils import generate_version_info
from numba.core import config
from numba.core import types, errors
from numba.misc.special import (
from numba.core.errors import *
import numba.core.types as types
from numba.core.types import *
from numba.core.decorators import (cfunc, jit, njit, stencil,
from numba.np.ufunc import (vectorize, guvectorize, threading_layer,
from numba.np.numpy_support import carray, farray, from_dtype
from numba import experimental
import numba.core.withcontexts
from numba.core.withcontexts import objmode_context as objmode
from numba.core.withcontexts import parallel_chunksize
import numba.core.target_extension
import numba.typed
import llvmlite
def _try_enable_svml():
    """
    Tries to enable SVML if configuration permits use and the library is found.
    """
    if not config.DISABLE_INTEL_SVML:
        try:
            if sys.platform.startswith('linux'):
                llvmlite.binding.load_library_permanently('libsvml.so')
            elif sys.platform.startswith('darwin'):
                llvmlite.binding.load_library_permanently('libsvml.dylib')
            elif sys.platform.startswith('win'):
                llvmlite.binding.load_library_permanently('svml_dispmd')
            else:
                return False
            try:
                if not getattr(llvmlite.binding.targets, 'has_svml')():
                    return False
            except AttributeError:
                if platform.machine() == 'x86_64' and config.DEBUG:
                    msg = 'SVML was found but llvmlite >= 0.23.2 is needed to support it.'
                    warnings.warn(msg)
                return False
            llvmlite.binding.set_option('SVML', '-vector-library=SVML')
            return True
        except:
            if platform.machine() == 'x86_64' and config.DEBUG:
                warnings.warn('SVML was not found/could not be loaded.')
    return False