import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_item_nan(self):
    cat = Categorical([1, 2, 3])
    cat[1] = np.nan
    exp = Categorical([1, np.nan, 3], categories=[1, 2, 3])
    tm.assert_categorical_equal(cat, exp)