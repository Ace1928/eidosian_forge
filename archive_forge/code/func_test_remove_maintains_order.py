import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_remove_maintains_order(self):
    ci = CategoricalIndex(list('abcdda'), categories=list('abcd'))
    result = ci.reorder_categories(['d', 'c', 'b', 'a'], ordered=True)
    tm.assert_index_equal(result, CategoricalIndex(list('abcdda'), categories=list('dcba'), ordered=True))
    result = result.remove_categories(['c'])
    tm.assert_index_equal(result, CategoricalIndex(['a', 'b', np.nan, 'd', 'd', 'a'], categories=list('dba'), ordered=True))