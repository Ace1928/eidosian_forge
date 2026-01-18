from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('value,float_format,expected', [(0.19999, '%.3f', 'gh21625_expected_output'), (100.0, '%.0f', 'gh22270_expected_output')])
def test_to_html_float_format_no_fixed_width(value, float_format, expected, datapath):
    df = DataFrame({'x': [value]})
    expected = expected_html(datapath, expected)
    result = df.to_html(float_format=float_format)
    assert result == expected