from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,kwargs,expected', [('foo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n', {'index_col': 0, 'names': ['index', 'A', 'B', 'C', 'D']}, DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], index=Index(['foo', 'bar', 'baz', 'qux', 'foo2', 'bar2'], name='index'), columns=['A', 'B', 'C', 'D'])), ('foo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n', {'index_col': [0, 1], 'names': ['index1', 'index2', 'A', 'B', 'C', 'D']}, DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], index=MultiIndex.from_tuples([('foo', 'one'), ('foo', 'two'), ('foo', 'three'), ('bar', 'one'), ('bar', 'two')], names=['index1', 'index2']), columns=['A', 'B', 'C', 'D']))])
def test_pass_names_with_index(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)