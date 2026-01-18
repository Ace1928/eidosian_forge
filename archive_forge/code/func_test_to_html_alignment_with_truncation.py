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
@pytest.mark.parametrize('column_index, column_type', [(Index(np.arange(8)), 'unnamed_standard'), (Index(np.arange(8), name='columns.name'), 'named_standard'), (MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']]), 'unnamed_multi'), (MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']], names=['foo', None, 'baz']), 'named_multi')])
@pytest.mark.parametrize('row_index, row_type', [(Index(np.arange(8)), 'unnamed_standard'), (Index(np.arange(8), name='index.name'), 'named_standard'), (MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']]), 'unnamed_multi'), (MultiIndex.from_product([['a', 'b'], ['c', 'd'], ['e', 'f']], names=['foo', None, 'baz']), 'named_multi')])
def test_to_html_alignment_with_truncation(datapath, row_index, row_type, column_index, column_type, index, header, index_names):
    df = DataFrame(np.arange(64).reshape(8, 8), index=row_index, columns=column_index)
    result = df.to_html(max_rows=4, max_cols=4, index=index, header=header, index_names=index_names)
    if not index:
        row_type = 'none'
    elif not index_names and row_type.startswith('named'):
        row_type = 'un' + row_type
    if not header:
        column_type = 'none'
    elif not index_names and column_type.startswith('named'):
        column_type = 'un' + column_type
    filename = 'trunc_df_index_' + row_type + '_columns_' + column_type
    expected = expected_html(datapath, filename)
    assert result == expected