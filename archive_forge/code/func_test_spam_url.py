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
@pytest.mark.network
@pytest.mark.single_cpu
def test_spam_url(self, httpserver, spam_data, flavor_read_html):
    with open(spam_data, encoding='utf-8') as f:
        httpserver.serve_content(content=f.read())
        df1 = flavor_read_html(httpserver.url, match='.*Water.*')
        df2 = flavor_read_html(httpserver.url, match='Unit')
    assert_framelist_equal(df1, df2)