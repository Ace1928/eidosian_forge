from __future__ import (absolute_import, division, print_function,
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from six.moves import xrange, zip
Returns the index in *sequence* of the missing data, identified by
        *miss*

        Parameters
        ----------
        sequence :
            sequence to evaluate
        miss :
            identifier of the missing data

        Returns
        -------
        where_miss: numpy.ndarray
            indices of the missing data
        