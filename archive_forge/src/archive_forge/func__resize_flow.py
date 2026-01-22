import numpy as np
from scipy import ndimage as ndi
from ..transform import pyramid_reduce
from ..util.dtype import _convert
def _resize_flow(flow, shape):
    """Rescale the values of the vector field (u, v) to the desired shape.

    The values of the output vector field are scaled to the new
    resolution.

    Parameters
    ----------
    flow : ndarray
        The motion field to be processed.
    shape : iterable
        Couple of integers representing the output shape.

    Returns
    -------
    rflow : ndarray
        The resized and rescaled motion field.

    """
    scale = [n / o for n, o in zip(shape, flow.shape[1:])]
    scale_factor = np.array(scale, dtype=flow.dtype)
    for _ in shape:
        scale_factor = scale_factor[..., np.newaxis]
    rflow = scale_factor * ndi.zoom(flow, [1] + scale, order=0, mode='nearest', prefilter=False)
    return rflow