import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas._config.config as cf
from pandas import Index
import pandas._testing as tm
def test_repr_is_valid_construction_code(self):
    idx = Index(['a', 'b'])
    res = eval(repr(idx))
    tm.assert_index_equal(res, idx)