from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_truncate_ndots(self):

    def getndots(s):
        return len(re.match('[^\\.]*(\\.*)', s).groups()[0])
    s = Series([0, 2, 3, 6])
    with option_context('display.max_rows', 2):
        strrepr = repr(s).replace('\n', '')
    assert getndots(strrepr) == 2
    s = Series([0, 100, 200, 400])
    with option_context('display.max_rows', 2):
        strrepr = repr(s).replace('\n', '')
    assert getndots(strrepr) == 3