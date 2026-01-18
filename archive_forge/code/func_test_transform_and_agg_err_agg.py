from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
def test_transform_and_agg_err_agg(axis, float_frame):
    msg = 'cannot combine transform and aggregation operations'
    with pytest.raises(ValueError, match=msg):
        with np.errstate(all='ignore'):
            float_frame.agg(['max', 'sqrt'], axis=axis)