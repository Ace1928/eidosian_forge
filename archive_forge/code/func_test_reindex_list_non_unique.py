import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_reindex_list_non_unique(self):
    msg = 'cannot reindex on an axis with duplicate labels'
    ci = CategoricalIndex(['a', 'b', 'c', 'a'])
    with pytest.raises(ValueError, match=msg):
        ci.reindex(['a', 'c'])