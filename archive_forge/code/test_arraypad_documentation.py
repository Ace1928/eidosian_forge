import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs

        Assert that 'wrap' pads only with multiples of the original area if
        the pad width is larger than the original array.
        