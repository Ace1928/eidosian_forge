import warnings
import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
def _check_size_footprint_structure(ndim, size, footprint, structure, stacklevel=3, force_footprint=False):
    if structure is None and footprint is None:
        if size is None:
            raise RuntimeError('no footprint or filter size provided')
        sizes = _util._fix_sequence_arg(size, ndim, 'size', int)
        if force_footprint:
            return (None, cupy.ones(sizes, bool), None)
        return (sizes, None, None)
    if size is not None:
        warnings.warn('ignoring size because {} is set'.format('structure' if footprint is None else 'footprint'), UserWarning, stacklevel=stacklevel + 1)
    if footprint is not None:
        footprint = cupy.array(footprint, bool, True, 'C')
        if not footprint.any():
            raise ValueError('all-zero footprint is not supported')
    if structure is None:
        if not force_footprint and footprint.all():
            if footprint.ndim != ndim:
                raise RuntimeError('size must have length equal to input rank')
            return (footprint.shape, None, None)
        return (None, footprint, None)
    structure = cupy.ascontiguousarray(structure)
    if footprint is None:
        footprint = cupy.ones(structure.shape, bool)
    return (None, footprint, structure)