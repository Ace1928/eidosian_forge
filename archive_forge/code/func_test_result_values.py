import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_result_values(self):
    tgt = [np.percentile(d, 28) for d in _rdat]
    res = np.nanpercentile(_ndat, 28, axis=1)
    assert_almost_equal(res, tgt)
    tgt = np.transpose([np.percentile(d, (28, 98)) for d in _rdat])
    res = np.nanpercentile(_ndat, (28, 98), axis=1)
    assert_almost_equal(res, tgt)