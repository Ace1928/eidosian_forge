from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_to_html_with_column_specific_col_space_raises():
    df = DataFrame(np.random.default_rng(2).random(size=(3, 3)), columns=['a', 'b', 'c'])
    msg = 'Col_space length\\(\\d+\\) should match DataFrame number of columns\\(\\d+\\)'
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space=[30, 40])
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space=[30, 40, 50, 60])
    msg = 'unknown column'
    with pytest.raises(ValueError, match=msg):
        df.to_html(col_space={'a': 'foo', 'b': 23, 'd': 34})