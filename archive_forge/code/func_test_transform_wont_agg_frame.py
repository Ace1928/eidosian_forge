from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func', [['max', 'min'], ['max', 'sqrt']])
def test_transform_wont_agg_frame(axis, float_frame, func):
    msg = 'Function did not transform'
    with pytest.raises(ValueError, match=msg):
        float_frame.transform(func, axis=axis)