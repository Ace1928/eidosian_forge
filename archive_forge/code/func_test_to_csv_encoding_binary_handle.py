import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('mode', ['wb', 'w'])
def test_to_csv_encoding_binary_handle(self, mode):
    """
        Binary file objects should honor a specified encoding.

        GH 23854 and GH 13068 with binary handles
        """
    content = 'a, b, 🐟'.encode('utf-8-sig')
    buffer = io.BytesIO(content)
    df = pd.read_csv(buffer, encoding='utf-8-sig')
    buffer = io.BytesIO()
    df.to_csv(buffer, mode=mode, encoding='utf-8-sig', index=False)
    buffer.seek(0)
    assert buffer.getvalue().startswith(content)
    with tm.ensure_clean() as path:
        with open(path, 'w+b') as handle:
            DataFrame().to_csv(handle, mode=mode, encoding='utf-8-sig')
            handle.seek(0)
            assert handle.read().startswith(b'\xef\xbb\xbf""')