from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
@pytest.mark.parametrize('sep', [',', '\t'])
@pytest.mark.parametrize('encoding', ['utf-16', 'utf-16le', 'utf-16be'])
def test_utf16_bom_skiprows(all_parsers, sep, encoding):
    parser = all_parsers
    data = 'skip this\nskip this too\nA,B,C\n1,2,3\n4,5,6'.replace(',', sep)
    path = f'__{uuid.uuid4()}__.csv'
    kwargs = {'sep': sep, 'skiprows': 2}
    utf8 = 'utf-8'
    with tm.ensure_clean(path) as path:
        bytes_data = data.encode(encoding)
        with open(path, 'wb') as f:
            f.write(bytes_data)
        with TextIOWrapper(BytesIO(data.encode(utf8)), encoding=utf8) as bytes_buffer:
            result = parser.read_csv(path, encoding=encoding, **kwargs)
            expected = parser.read_csv(bytes_buffer, encoding=utf8, **kwargs)
        tm.assert_frame_equal(result, expected)