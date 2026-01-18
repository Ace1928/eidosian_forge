import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_take_indexer_type(self):
    integer_index = Index([0, 1, 2, 3])
    scalar_index = 1
    msg = 'Expected indices to be array-like'
    with pytest.raises(TypeError, match=msg):
        integer_index.take(scalar_index)