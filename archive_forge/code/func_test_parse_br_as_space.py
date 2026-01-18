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
def test_parse_br_as_space(self, flavor_read_html):
    result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <th>A</th>\n                </tr>\n                <tr>\n                    <td>word1<br>word2</td>\n                </tr>\n            </table>\n        '))[0]
    expected = DataFrame(data=[['word1 word2']], columns=['A'])
    tm.assert_frame_equal(result, expected)