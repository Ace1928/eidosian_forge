from collections import namedtuple
import numpy as np
from ...util import dtype as dtypes
from ...exposure import is_low_contrast
from ..._shared.utils import warn
from math import floor, ceil
Display all images in the collection.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The `Figure` object returned by `plt.subplots`.
    