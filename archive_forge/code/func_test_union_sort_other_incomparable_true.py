from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_union_sort_other_incomparable_true(self):
    idx = Index([1, pd.Timestamp('2000')])
    with pytest.raises(TypeError, match='.*'):
        idx.union(idx[:1], sort=True)