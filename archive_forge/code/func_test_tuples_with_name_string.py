from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_tuples_with_name_string():
    li = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    msg = 'Names should be list-like for a MultiIndex'
    with pytest.raises(ValueError, match=msg):
        Index(li, name='abc')
    with pytest.raises(ValueError, match=msg):
        Index(li, name='a')