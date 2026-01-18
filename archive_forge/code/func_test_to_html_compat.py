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
def test_to_html_compat(self, flavor_read_html):
    df = DataFrame(np.random.default_rng(2).random((4, 3)), columns=pd.Index(list('abc'), dtype=object)).map('{:.3f}'.format).astype(float)
    out = df.to_html()
    res = flavor_read_html(StringIO(out), attrs={'class': 'dataframe'}, index_col=0)[0]
    tm.assert_frame_equal(res, df)