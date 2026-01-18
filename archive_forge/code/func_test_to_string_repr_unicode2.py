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
def test_to_string_repr_unicode2(self):
    idx = Index(['abc', 'σa', 'aegdvg'])
    ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
    rs = repr(ser).split('\n')
    line_len = len(rs[0])
    for line in rs[1:]:
        try:
            line = line.decode(get_option('display.encoding'))
        except AttributeError:
            pass
        if not line.startswith('dtype:'):
            assert len(line) == line_len