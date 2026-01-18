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
@pytest.mark.parametrize('sparsify,expected', [(True, 'truncate_multi_index'), (False, 'truncate_multi_index_sparse_off')])
def test_to_html_truncate_multi_index(sparsify, expected, datapath):
    arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
    df = DataFrame(index=arrays, columns=arrays)
    result = df.to_html(max_rows=7, max_cols=7, sparsify=sparsify)
    expected = expected_html(datapath, expected)
    assert result == expected