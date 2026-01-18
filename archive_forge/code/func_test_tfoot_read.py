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
def test_tfoot_read(self, flavor_read_html):
    """
        Make sure that read_html reads tfoot, containing td or th.
        Ignores empty tfoot
        """
    data_template = '<table>\n            <thead>\n                <tr>\n                    <th>A</th>\n                    <th>B</th>\n                </tr>\n            </thead>\n            <tbody>\n                <tr>\n                    <td>bodyA</td>\n                    <td>bodyB</td>\n                </tr>\n            </tbody>\n            <tfoot>\n                {footer}\n            </tfoot>\n        </table>'
    expected1 = DataFrame(data=[['bodyA', 'bodyB']], columns=['A', 'B'])
    expected2 = DataFrame(data=[['bodyA', 'bodyB'], ['footA', 'footB']], columns=['A', 'B'])
    data1 = data_template.format(footer='')
    data2 = data_template.format(footer='<tr><td>footA</td><th>footB</th></tr>')
    result1 = flavor_read_html(StringIO(data1))[0]
    result2 = flavor_read_html(StringIO(data2))[0]
    tm.assert_frame_equal(result1, expected1)
    tm.assert_frame_equal(result2, expected2)