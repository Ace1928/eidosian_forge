import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_type_pickle():
    import pickle
    np._ScaledFloatTestDType = SF
    s = pickle.dumps(SF)
    res = pickle.loads(s)
    assert res is SF
    del np._ScaledFloatTestDType