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
def test_show_dimensions(self):
    s = Series(range(5))
    assert 'Length' not in repr(s)
    with option_context('display.max_rows', 4):
        assert 'Length' in repr(s)
    with option_context('display.show_dimensions', True):
        assert 'Length' in repr(s)
    with option_context('display.max_rows', 4, 'display.show_dimensions', False):
        assert 'Length' not in repr(s)