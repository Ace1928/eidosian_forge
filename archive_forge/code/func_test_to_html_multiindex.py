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
@pytest.mark.parametrize('columns,justify,expected', [(MultiIndex.from_arrays([np.arange(2).repeat(2), np.mod(range(4), 2)], names=['CL0', 'CL1']), 'left', 'multiindex_1'), (MultiIndex.from_arrays([np.arange(4), np.mod(range(4), 2)]), 'right', 'multiindex_2')])
def test_to_html_multiindex(columns, justify, expected, datapath):
    df = DataFrame([list('abcd'), list('efgh')], columns=columns)
    result = df.to_html(justify=justify)
    expected = expected_html(datapath, expected)
    assert result == expected