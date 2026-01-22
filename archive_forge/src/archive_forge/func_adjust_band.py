from collections import OrderedDict
from itertools import zip_longest
import logging
import warnings
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import guard_transform
def adjust_band(band, kind=None):
    """Adjust a band to be between 0 and 1.

    Parameters
    ----------
    band : array, shape (height, width)
        A band of a raster object.
    kind : str
        An unused option. For now, there is only one option ('linear').

    Returns
    -------
    band_normed : array, shape (height, width)
        An adjusted version of the input band.

    """
    imin = np.float64(np.nanmin(band))
    imax = np.float64(np.nanmax(band))
    return (band - imin) / (imax - imin)