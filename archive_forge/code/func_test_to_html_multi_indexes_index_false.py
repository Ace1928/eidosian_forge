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
def test_to_html_multi_indexes_index_false(datapath):
    df = DataFrame({'a': range(10), 'b': range(10, 20), 'c': range(10, 20), 'd': range(10, 20)})
    df.columns = MultiIndex.from_product([['a', 'b'], ['c', 'd']])
    df.index = MultiIndex.from_product([['a', 'b'], ['c', 'd', 'e', 'f', 'g']])
    result = df.to_html(index=False)
    expected = expected_html(datapath, 'gh22579_expected_output')
    assert result == expected