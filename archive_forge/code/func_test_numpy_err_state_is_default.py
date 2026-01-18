import os
import pytest
from pandas import (
import pandas._testing as tm
def test_numpy_err_state_is_default():
    expected = {'over': 'warn', 'divide': 'warn', 'invalid': 'warn', 'under': 'ignore'}
    import numpy as np
    assert np.geterr() == expected