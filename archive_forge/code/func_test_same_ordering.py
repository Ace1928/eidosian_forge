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
def test_same_ordering(datapath):
    pytest.importorskip('bs4')
    pytest.importorskip('lxml')
    pytest.importorskip('html5lib')
    filename = datapath('io', 'data', 'html', 'valid_markup.html')
    dfs_lxml = read_html(filename, index_col=0, flavor=['lxml'])
    dfs_bs4 = read_html(filename, index_col=0, flavor=['bs4'])
    assert_framelist_equal(dfs_lxml, dfs_bs4)