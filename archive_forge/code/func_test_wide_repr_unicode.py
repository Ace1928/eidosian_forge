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
def test_wide_repr_unicode(self):
    with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
        max_cols = 20
        df = DataFrame([['a' * 25] * 10] * (max_cols - 1))
        with option_context('display.expand_frame_repr', False):
            rep_str = repr(df)
        with option_context('display.expand_frame_repr', True):
            wide_repr = repr(df)
        assert rep_str != wide_repr
        with option_context('display.width', 150):
            wider_repr = repr(df)
            assert len(wider_repr) < len(wide_repr)