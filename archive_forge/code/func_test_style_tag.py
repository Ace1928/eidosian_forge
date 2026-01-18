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
def test_style_tag(self, flavor_read_html):
    data = '\n        <table>\n            <tr>\n                <th>\n                    <style>.style</style>\n                    A\n                    </th>\n                <th>B</th>\n            </tr>\n            <tr>\n                <td>A1</td>\n                <td>B1</td>\n            </tr>\n            <tr>\n                <td>A2</td>\n                <td>B2</td>\n            </tr>\n        </table>\n        '
    result = flavor_read_html(StringIO(data))[0]
    expected = DataFrame(data=[['A1', 'B1'], ['A2', 'B2']], columns=['A', 'B'])
    tm.assert_frame_equal(result, expected)