import itertools
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import draw
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.utils import _supported_float_type
from skimage.measure import (
Compare two moments arrays.

    Compares only values in the upper-left triangle of m1, m2 since
    values below the diagonal exceed the specified order and are not computed
    when the analytical computation is used.

    Also, there the first-order central moments will be exactly zero with the
    analytical calculation, but will not be zero due to limited floating point
    precision when using a numerical computation. Here we just specify the
    tolerance as a fraction of the maximum absolute value in the moments array.
    