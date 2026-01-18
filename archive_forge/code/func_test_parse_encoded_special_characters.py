from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16', 'utf-16-be', 'utf-16-le', 'utf-32'])
def test_parse_encoded_special_characters(encoding):
    data = 'a\tb\n：foo\t0\nbar\t1\nbaz\t2'
    encoded_data = BytesIO(data.encode(encoding))
    result = read_csv(encoded_data, delimiter='\t', encoding=encoding)
    expected = DataFrame(data=[['：foo', 0], ['bar', 1], ['baz', 2]], columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)