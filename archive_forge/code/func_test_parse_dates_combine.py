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
def test_parse_dates_combine(self, flavor_read_html):
    raw_dates = Series(date_range('1/1/2001', periods=10))
    df = DataFrame({'date': raw_dates.map(lambda x: str(x.date())), 'time': raw_dates.map(lambda x: str(x.time()))})
    res = flavor_read_html(StringIO(df.to_html()), parse_dates={'datetime': [1, 2]}, index_col=1)
    newdf = DataFrame({'datetime': raw_dates})
    tm.assert_frame_equal(newdf, res[0])