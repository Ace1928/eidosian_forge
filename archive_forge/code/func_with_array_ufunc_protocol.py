import functools
import numpy as _np
from . import numpy as mx_np  # pylint: disable=reimported
from .numpy.multiarray import _NUMPY_ARRAY_FUNCTION_DICT, _NUMPY_ARRAY_UFUNC_DICT
def with_array_ufunc_protocol(func):
    """A decorator for functions that expect array ufunc protocol.
    The decorated function only runs when NumPy version >= 1.15."""
    from distutils.version import LooseVersion
    cur_np_ver = LooseVersion(_np.__version__)
    np_1_15_ver = LooseVersion('1.15')

    @functools.wraps(func)
    def _run_with_array_ufunc_proto(*args, **kwargs):
        if cur_np_ver >= np_1_15_ver:
            try:
                func(*args, **kwargs)
            except Exception as e:
                raise RuntimeError('Running function {} with NumPy array ufunc protocol failed with exception {}'.format(func.__name__, str(e)))
    return _run_with_array_ufunc_proto