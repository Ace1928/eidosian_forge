import math
from functools import partial
import numpy as np
import param
from ..core import NdOverlay, Overlay
from ..element.chart import Area
from .resample import ResampleOperation1D
def _nth_point(x, y, n_out):
    """
    Downsampling by selecting every n-th datapoint

    Args:
        x (np.ndarray): The x-values of the data.
        y (np.ndarray): The y-values of the data.
        n_out (int): The number of output points.
    Returns:
        np.array: The indexes of the selected datapoints.
    """
    n_samples = len(x)
    return np.arange(0, n_samples, max(1, math.ceil(n_samples / n_out)))