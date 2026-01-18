from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bad_line_func', [lambda x: ['foo', 'bar'], lambda x: x[:2]])
@pytest.mark.parametrize('sep', [',', '111'])
def test_on_bad_lines_callable_iterator_true(python_parser_only, bad_line_func, sep):
    parser = python_parser_only
    data = f'\n0{sep}1\nhi{sep}there\nfoo{sep}bar{sep}baz\ngood{sep}bye\n'
    bad_sio = StringIO(data)
    result_iter = parser.read_csv(bad_sio, on_bad_lines=bad_line_func, chunksize=1, iterator=True, sep=sep)
    expecteds = [{'0': 'hi', '1': 'there'}, {'0': 'foo', '1': 'bar'}, {'0': 'good', '1': 'bye'}]
    for i, (result, expected) in enumerate(zip(result_iter, expecteds)):
        expected = DataFrame(expected, index=range(i, i + 1))
        tm.assert_frame_equal(result, expected)