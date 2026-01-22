from numpy.ma import (
import numpy.ma as ma
import warnings
import numpy as np
from numpy import (
from numpy.core.records import (
def _get_fieldmask(self):
    mdescr = [(n, '|b1') for n in self.dtype.names]
    fdmask = np.empty(self.shape, dtype=mdescr)
    fdmask.flat = tuple([False] * len(mdescr))
    return fdmask