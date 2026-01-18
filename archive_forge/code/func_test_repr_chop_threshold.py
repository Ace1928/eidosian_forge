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
def test_repr_chop_threshold(self):
    df = DataFrame([[0.1, 0.5], [0.5, -0.1]])
    reset_option('display.chop_threshold')
    assert repr(df) == '     0    1\n0  0.1  0.5\n1  0.5 -0.1'
    with option_context('display.chop_threshold', 0.2):
        assert repr(df) == '     0    1\n0  0.0  0.5\n1  0.5  0.0'
    with option_context('display.chop_threshold', 0.6):
        assert repr(df) == '     0    1\n0  0.0  0.0\n1  0.0  0.0'
    with option_context('display.chop_threshold', None):
        assert repr(df) == '     0    1\n0  0.1  0.5\n1  0.5 -0.1'