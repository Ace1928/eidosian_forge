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
def test_parse_header_of_non_string_column(self, flavor_read_html):
    result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <td>S</td>\n                    <td>I</td>\n                </tr>\n                <tr>\n                    <td>text</td>\n                    <td>1944</td>\n                </tr>\n            </table>\n        '), header=0)[0]
    expected = DataFrame([['text', 1944]], columns=('S', 'I'))
    tm.assert_frame_equal(result, expected)