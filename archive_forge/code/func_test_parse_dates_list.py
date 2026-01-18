from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
def test_parse_dates_list(self, flavor_read_html):
    df = DataFrame({'date': date_range('1/1/2001', periods=10)})
    expected = df.to_html()
    res = flavor_read_html(StringIO(expected), parse_dates=[1], index_col=0)
    tm.assert_frame_equal(df, res[0])
    res = flavor_read_html(StringIO(expected), parse_dates=['date'], index_col=0)
    tm.assert_frame_equal(df, res[0])