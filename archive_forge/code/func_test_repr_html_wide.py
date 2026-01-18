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
def test_repr_html_wide(self):
    max_cols = 20
    df = DataFrame([['a' * 25] * (max_cols - 1)] * 10)
    with option_context('display.max_rows', 60, 'display.max_columns', 20):
        assert '...' not in df._repr_html_()
    wide_df = DataFrame([['a' * 25] * (max_cols + 1)] * 10)
    with option_context('display.max_rows', 60, 'display.max_columns', 20):
        assert '...' in wide_df._repr_html_()