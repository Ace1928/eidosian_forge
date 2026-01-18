import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_standard_deviation01():
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([], type)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, 'Mean of empty slice')
                output = ndimage.standard_deviation(input)
            assert_(np.isnan(output))