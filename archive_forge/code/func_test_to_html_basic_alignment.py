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
@pytest.mark.parametrize('index_names', [True, False])
@pytest.mark.parametrize('header', [True, False])
@pytest.mark.parametrize('index', [True, False])
@pytest.mark.parametrize('column_index, column_type', [(Index([0, 1]), 'unnamed_standard'), (Index([0, 1], name='columns.name'), 'named_standard'), (MultiIndex.from_product([['a'], ['b', 'c']]), 'unnamed_multi'), (MultiIndex.from_product([['a'], ['b', 'c']], names=['columns.name.0', 'columns.name.1']), 'named_multi')])
@pytest.mark.parametrize('row_index, row_type', [(Index([0, 1]), 'unnamed_standard'), (Index([0, 1], name='index.name'), 'named_standard'), (MultiIndex.from_product([['a'], ['b', 'c']]), 'unnamed_multi'), (MultiIndex.from_product([['a'], ['b', 'c']], names=['index.name.0', 'index.name.1']), 'named_multi')])
def test_to_html_basic_alignment(datapath, row_index, row_type, column_index, column_type, index, header, index_names):
    df = DataFrame(np.zeros((2, 2), dtype=int), index=row_index, columns=column_index)
    result = df.to_html(index=index, header=header, index_names=index_names)
    if not index:
        row_type = 'none'
    elif not index_names and row_type.startswith('named'):
        row_type = 'un' + row_type
    if not header:
        column_type = 'none'
    elif not index_names and column_type.startswith('named'):
        column_type = 'un' + column_type
    filename = 'index_' + row_type + '_columns_' + column_type
    expected = expected_html(datapath, filename)
    assert result == expected