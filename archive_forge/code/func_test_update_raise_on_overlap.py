import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_update_raise_on_overlap(self):
    df = DataFrame([[1.5, 1, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]])
    other = DataFrame([[2.0, np.nan], [np.nan, 7]], index=[1, 3], columns=[1, 2])
    with pytest.raises(ValueError, match='Data overlaps'):
        df.update(other, errors='raise')