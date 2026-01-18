import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
@staticmethod
def z_score(data2d, axis=1):
    """Standarize the mean and variance of the data axis

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        normalized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.
        """
    if axis == 1:
        z_scored = data2d
    else:
        z_scored = data2d.T
    z_scored = (z_scored - z_scored.mean()) / z_scored.std()
    if axis == 1:
        return z_scored
    else:
        return z_scored.T