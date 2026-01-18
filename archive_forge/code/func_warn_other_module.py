import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def warn_other_module():

    def warn(arr):
        warnings.warn('Some warning', stacklevel=2)
        return arr
    np.apply_along_axis(warn, 0, [0])