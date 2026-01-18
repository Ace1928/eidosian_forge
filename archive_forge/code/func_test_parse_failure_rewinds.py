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
def test_parse_failure_rewinds(self, flavor_read_html):

    class MockFile:

        def __init__(self, data) -> None:
            self.data = data
            self.at_end = False

        def read(self, size=None):
            data = '' if self.at_end else self.data
            self.at_end = True
            return data

        def seek(self, offset):
            self.at_end = False

        def seekable(self):
            return True

        def __next__(self):
            ...

        def __iter__(self) -> Iterator:
            return self
    good = MockFile('<table><tr><td>spam<br />eggs</td></tr></table>')
    bad = MockFile('<table><tr><td>spam<foobr />eggs</td></tr></table>')
    assert flavor_read_html(good)
    assert flavor_read_html(bad)