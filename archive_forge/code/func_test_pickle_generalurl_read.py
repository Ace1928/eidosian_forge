from __future__ import annotations
from array import array
import bz2
import datetime
import functools
from functools import partial
import gzip
import io
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any
import uuid
import zipfile
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.compat.compressors import flatten_buffer
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.generate_legacy_storage_files import create_pickle_data
import pandas.io.common as icom
from pandas.tseries.offsets import (
@pytest.mark.parametrize('mockurl', ['http://url.com', 'ftp://test.com', 'http://gzip.com'])
def test_pickle_generalurl_read(monkeypatch, mockurl):

    def python_pickler(obj, path):
        with open(path, 'wb') as fh:
            pickle.dump(obj, fh, protocol=-1)

    class MockReadResponse:

        def __init__(self, path) -> None:
            self.file = open(path, 'rb')
            if 'gzip' in path:
                self.headers = {'Content-Encoding': 'gzip'}
            else:
                self.headers = {'Content-Encoding': ''}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def read(self):
            return self.file.read()

        def close(self):
            return self.file.close()
    with tm.ensure_clean() as path:

        def mock_urlopen_read(*args, **kwargs):
            return MockReadResponse(path)
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        python_pickler(df, path)
        monkeypatch.setattr('urllib.request.urlopen', mock_urlopen_read)
        result = pd.read_pickle(mockurl)
        tm.assert_frame_equal(df, result)