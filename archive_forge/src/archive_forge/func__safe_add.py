from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def _safe_add(df):

    def is_ok(s):
        return issubclass(s.dtype.type, (np.integer, np.floating)) and s.dtype != 'uint8'
    return DataFrame(dict(((c, s + 1) if is_ok(s) else (c, s) for c, s in df.items())))