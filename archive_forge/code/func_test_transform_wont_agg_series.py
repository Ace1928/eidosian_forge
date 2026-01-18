from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func', [['min', 'max'], ['sqrt', 'max']])
def test_transform_wont_agg_series(string_series, func):
    msg = 'Function did not transform'
    with pytest.raises(ValueError, match=msg):
        string_series.transform(func)