import builtins
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
def test_object_NA_raises_with_skipna_false(bool_agg_func):
    ser = Series([pd.NA], dtype=object)
    with pytest.raises(TypeError, match='boolean value of NA is ambiguous'):
        ser.groupby([1]).agg(bool_agg_func, skipna=False)