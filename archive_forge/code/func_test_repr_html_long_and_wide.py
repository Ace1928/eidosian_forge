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
def test_repr_html_long_and_wide(self):
    max_cols = 20
    max_rows = 60
    h, w = (max_rows - 1, max_cols - 1)
    df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
    with option_context('display.max_rows', 60, 'display.max_columns', 20):
        assert '...' not in df._repr_html_()
    h, w = (max_rows + 1, max_cols + 1)
    df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
    with option_context('display.max_rows', 60, 'display.max_columns', 20):
        assert '...' in df._repr_html_()