from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_truthy(self):
    df = DataFrame({'a': [True, True]})
    r = df.replace([np.inf, -np.inf], np.nan)
    e = df
    tm.assert_frame_equal(r, e)