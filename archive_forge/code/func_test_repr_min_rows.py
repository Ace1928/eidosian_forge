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
def test_repr_min_rows(self):
    s = Series(range(20))
    assert '..' not in repr(s)
    s = Series(range(61))
    assert '..' in repr(s)
    with option_context('display.max_rows', 10, 'display.min_rows', 4):
        assert '..' in repr(s)
        assert '2  ' not in repr(s)
    with option_context('display.max_rows', 12, 'display.min_rows', None):
        assert '5      5' in repr(s)
    with option_context('display.max_rows', 10, 'display.min_rows', 12):
        assert '5      5' not in repr(s)
    with option_context('display.max_rows', None, 'display.min_rows', 12):
        assert '..' not in repr(s)