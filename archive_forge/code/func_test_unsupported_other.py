import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.feather_format import read_feather, to_feather  # isort:skip
def test_unsupported_other(self):
    df = pd.DataFrame({'a': ['a', 1, 2.0]})
    self.check_external_error_on_write(df)