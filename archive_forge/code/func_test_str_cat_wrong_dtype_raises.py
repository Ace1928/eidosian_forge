import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.mark.parametrize('data', [[1, 2, 3], [0.1, 0.2, 0.3], [1, 2, 'b']], ids=['integers', 'floats', 'mixed'])
@pytest.mark.parametrize('box', [Series, Index, list, lambda x: np.array(x, dtype=object)], ids=['Series', 'Index', 'list', 'np.array'])
def test_str_cat_wrong_dtype_raises(box, data):
    s = Series(['a', 'b', 'c'])
    t = box(data)
    msg = 'Concatenation requires list-likes containing only strings.*'
    with pytest.raises(TypeError, match=msg):
        s.str.cat(t, join='outer', na_rep='-')