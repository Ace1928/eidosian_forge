import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
@pytest.mark.parametrize('method', ['min', 'max'])
def test_numeric_only_min_max_raises(self, method):
    cat = Categorical([np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True)
    with pytest.raises(TypeError, match='.* got an unexpected keyword'):
        getattr(cat, method)(numeric_only=True)