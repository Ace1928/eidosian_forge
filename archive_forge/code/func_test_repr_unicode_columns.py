from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_unicode_columns(self):
    df = DataFrame({'א': [1, 2, 3], 'ב': [4, 5, 6], 'c': [7, 8, 9]})
    repr(df.columns)