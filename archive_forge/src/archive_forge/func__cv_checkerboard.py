import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_checkerboard(image_size, square_size, dtype=np.float64):
    """Generates a checkerboard level set function.

    According to Pascal Getreuer, such a level set function has fast
    convergence.
    """
    yv = np.arange(image_size[0], dtype=dtype).reshape(image_size[0], 1)
    xv = np.arange(image_size[1], dtype=dtype)
    sf = np.pi / square_size
    xv *= sf
    yv *= sf
    return np.sin(yv) * np.sin(xv)