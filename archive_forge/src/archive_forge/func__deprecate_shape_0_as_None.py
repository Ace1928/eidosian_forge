import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import os_fspath
from .arrayprint import _get_legacy_print_mode
def _deprecate_shape_0_as_None(shape):
    if shape == 0:
        warnings.warn('Passing `shape=0` to have the shape be inferred is deprecated, and in future will be equivalent to `shape=(0,)`. To infer the shape and suppress this warning, pass `shape=None` instead.', FutureWarning, stacklevel=3)
        return None
    else:
        return shape