from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,kwargs,expected', [('a\n1', {}, DataFrame({'a': [1]})), ('"a"\n1', {'quotechar': '"'}, DataFrame({'a': [1]})), ('b\n1', {'names': ['a']}, DataFrame({'a': ['b', '1']})), ('\n1', {'names': ['a'], 'skip_blank_lines': True}, DataFrame({'a': [1]})), ('\n1', {'names': ['a'], 'skip_blank_lines': False}, DataFrame({'a': [np.nan, 1]}))])
def test_utf8_bom(all_parsers, data, kwargs, expected, request):
    parser = all_parsers
    bom = '\ufeff'
    utf8 = 'utf-8'

    def _encode_data_with_bom(_data):
        bom_data = (bom + _data).encode(utf8)
        return BytesIO(bom_data)
    if parser.engine == 'pyarrow' and data == '\n1' and kwargs.get('skip_blank_lines', True):
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    result = parser.read_csv(_encode_data_with_bom(data), encoding=utf8, **kwargs)
    tm.assert_frame_equal(result, expected)