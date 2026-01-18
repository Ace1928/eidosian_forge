import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_unique_all_sparse():
    arr = SparseArray([0, 0])
    result = arr.unique()
    expected = SparseArray([0])
    tm.assert_sp_array_equal(result, expected)