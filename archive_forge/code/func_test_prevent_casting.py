import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_prevent_casting(self, simple_index):
    index = simple_index
    result = index.astype('O')
    assert result.dtype == np.object_