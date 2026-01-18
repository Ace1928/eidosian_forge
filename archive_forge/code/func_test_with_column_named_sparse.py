import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_with_column_named_sparse(self):
    df = pd.DataFrame({'sparse': pd.arrays.SparseArray([1, 2])})
    assert isinstance(df.sparse, pd.core.arrays.sparse.accessor.SparseFrameAccessor)