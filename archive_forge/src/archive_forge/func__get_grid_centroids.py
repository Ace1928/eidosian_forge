import math
from collections.abc import Iterable
from warnings import warn
import numpy as np
from numpy import random
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform
from .._shared import utils
from .._shared.filters import gaussian
from ..color import rgb2lab
from ..util import img_as_float, regular_grid
from ._slic import _enforce_label_connectivity_cython, _slic_cython
def _get_grid_centroids(image, n_centroids):
    """Find regularly spaced centroids on the image.

    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or
        multichannel.
    n_centroids : int
        The (approximate) number of centroids to be returned.

    Returns
    -------
    centroids : 2D ndarray
        The coordinates of the centroids with shape (~n_centroids, 3).
    steps : 1D ndarray
        The approximate distance between two seeds in all dimensions.

    """
    d, h, w = image.shape[:3]
    grid_z, grid_y, grid_x = np.mgrid[:d, :h, :w]
    slices = regular_grid(image.shape[:3], n_centroids)
    centroids_z = grid_z[slices].ravel()[..., np.newaxis]
    centroids_y = grid_y[slices].ravel()[..., np.newaxis]
    centroids_x = grid_x[slices].ravel()[..., np.newaxis]
    centroids = np.concatenate([centroids_z, centroids_y, centroids_x], axis=-1)
    steps = np.asarray([float(s.step) if s.step is not None else 1.0 for s in slices])
    return (centroids, steps)