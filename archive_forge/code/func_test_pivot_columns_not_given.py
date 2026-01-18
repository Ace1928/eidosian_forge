from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_columns_not_given(self):
    df = DataFrame({'a': [1], 'b': 1})
    with pytest.raises(TypeError, match='missing 1 required keyword-only argument'):
        df.pivot()