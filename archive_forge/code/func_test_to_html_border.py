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
@pytest.mark.parametrize('option,result,expected', [(None, lambda df: df.to_html(), '1'), (None, lambda df: df.to_html(border=2), '2'), (2, lambda df: df.to_html(), '2'), (2, lambda df: df._repr_html_(), '2')])
def test_to_html_border(option, result, expected):
    df = DataFrame({'A': [1, 2]})
    if option is None:
        result = result(df)
    else:
        with option_context('display.html.border', option):
            result = result(df)
    expected = f'border="{expected}"'
    assert expected in result