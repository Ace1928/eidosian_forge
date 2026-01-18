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
@pytest.mark.parametrize('float_format,expected', [('{:,.0f}'.format, '0   1,000\n1    test\ndtype: object'), ('{:.4f}'.format, '0   1000.0000\n1        test\ndtype: object')])
def test_repr_float_format_in_object_col(self, float_format, expected):
    df = Series([1000.0, 'test'])
    with option_context('display.float_format', float_format):
        result = repr(df)
    assert result == expected