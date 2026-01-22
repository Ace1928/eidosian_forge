import warnings
import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
def _check_nd_args(input, weights, mode, origin, wghts_name='filter weights'):
    _util._check_mode(mode)
    if weights.nbytes >= 1 << 31:
        raise RuntimeError('weights must be 2 GiB or less, use FFTs instead')
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError('{} array has incorrect shape'.format(wghts_name))
    origins = _util._fix_sequence_arg(origin, len(weight_dims), 'origin', int)
    for origin, width in zip(origins, weight_dims):
        _util._check_origin(origin, width)
    return (tuple(origins), _util._get_inttype(input))