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
def test_pivot_duplicates(self):
    data = DataFrame({'a': ['bar', 'bar', 'foo', 'foo', 'foo'], 'b': ['one', 'two', 'one', 'one', 'two'], 'c': [1.0, 2.0, 3.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match='duplicate entries'):
        data.pivot(index='a', columns='b', values='c')