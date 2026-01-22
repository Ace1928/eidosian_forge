import functools
import numpy as _np
from . import numpy as mx_np  # pylint: disable=reimported
from .numpy.multiarray import _NUMPY_ARRAY_FUNCTION_DICT, _NUMPY_ARRAY_UFUNC_DICT
@with_array_ufunc_protocol
def _register_array_ufunc():
    """Register NumPy array ufunc protocol.

    References
    ----------
    https://numpy.org/neps/nep-0013-ufunc-overrides.html
    """
    dup = _find_duplicate(_NUMPY_ARRAY_UFUNC_LIST)
    if dup is not None:
        raise ValueError('Duplicate operator name {} in _NUMPY_ARRAY_UFUNC_LIST'.format(dup))
    for op_name in _NUMPY_ARRAY_UFUNC_LIST:
        try:
            mx_np_op = getattr(mx_np, op_name)
            _NUMPY_ARRAY_UFUNC_DICT[op_name] = mx_np_op
        except AttributeError:
            raise AttributeError('mxnet.numpy does not have operator named {}'.format(op_name))