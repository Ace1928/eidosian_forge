import numpy as np
from scipy import ndimage as ndi
from ..transform import pyramid_reduce
from ..util.dtype import _convert
def _coarse_to_fine(I0, I1, solver, downscale=2, nlevel=10, min_size=16, dtype=np.float32):
    """Generic coarse to fine solver.

    Parameters
    ----------
    I0 : ndarray
        The first grayscale image of the sequence.
    I1 : ndarray
        The second grayscale image of the sequence.
    solver : callable
        The solver applied at each pyramid level.
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.
    dtype : dtype
        Output data type.

    Returns
    -------
    flow : ndarray
        The estimated optical flow components for each axis.

    """
    if I0.shape != I1.shape:
        raise ValueError('Input images should have the same shape')
    if np.dtype(dtype).char not in 'efdg':
        raise ValueError('Only floating point data type are valid for optical flow')
    pyramid = list(zip(_get_pyramid(_convert(I0, dtype), downscale, nlevel, min_size), _get_pyramid(_convert(I1, dtype), downscale, nlevel, min_size)))
    flow = np.zeros((pyramid[0][0].ndim,) + pyramid[0][0].shape, dtype=dtype)
    flow = solver(pyramid[0][0], pyramid[0][1], flow)
    for J0, J1 in pyramid[1:]:
        flow = solver(J0, J1, _resize_flow(flow, J0.shape))
    return flow