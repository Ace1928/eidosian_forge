import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
@pytest.mark.parametrize('nan_func,func', NANFUNCS.items(), ids=IDS)
def test_signature_match(self, nan_func, func):
    signature = self.get_signature(func)
    nan_signature = self.get_signature(nan_func)
    np.testing.assert_equal(signature, nan_signature)