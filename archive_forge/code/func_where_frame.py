from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.fixture(params=['default', 'float_string', 'mixed_float', 'mixed_int'])
def where_frame(request, float_string_frame, mixed_float_frame, mixed_int_frame):
    if request.param == 'default':
        return DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['A', 'B', 'C'])
    if request.param == 'float_string':
        return float_string_frame
    if request.param == 'mixed_float':
        return mixed_float_frame
    if request.param == 'mixed_int':
        return mixed_int_frame