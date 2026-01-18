from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_with_categorical_columns(self):
    data = [[4, 2], [3, 2], [4, 3]]
    cols = ['aaaaaaaaa', 'b']
    df = DataFrame(data, columns=cols)
    df_cat_cols = DataFrame(data, columns=CategoricalIndex(cols))
    assert df.to_string() == df_cat_cols.to_string()