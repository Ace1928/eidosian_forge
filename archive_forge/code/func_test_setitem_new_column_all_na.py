import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_new_column_all_na(self):
    mix = MultiIndex.from_tuples([('1a', '2a'), ('1a', '2b'), ('1a', '2c')])
    df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mix)
    s = Series({(1, 1): 1, (1, 2): 2})
    df['new'] = s
    assert df['new'].isna().all()