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
def test_wide_repr_multiindex_cols(self):
    with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
        max_cols = get_option('display.max_columns')
        midx = MultiIndex.from_arrays([['a' * 5] * 10] * 2)
        mcols = MultiIndex.from_arrays([['b' * 3] * (max_cols - 1)] * 2)
        df = DataFrame([['c' * 25] * (max_cols - 1)] * 10, index=midx, columns=mcols)
        df.index.names = ['Level 0', 'Level 1']
        with option_context('display.expand_frame_repr', False):
            rep_str = repr(df)
        with option_context('display.expand_frame_repr', True):
            wide_repr = repr(df)
        assert rep_str != wide_repr
    with option_context('display.width', 150, 'display.max_columns', 20):
        wider_repr = repr(df)
        assert len(wider_repr) < len(wide_repr)