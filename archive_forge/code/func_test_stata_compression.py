import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.mark.parametrize('to_infer', [True, False])
@pytest.mark.parametrize('read_infer', [True, False])
def test_stata_compression(compression_only, read_infer, to_infer, compression_to_extension):
    compression = compression_only
    ext = compression_to_extension[compression]
    filename = f'test.{ext}'
    df = DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
    df.index.name = 'index'
    to_compression = 'infer' if to_infer else compression
    read_compression = 'infer' if read_infer else compression
    with tm.ensure_clean(filename) as path:
        df.to_stata(path, compression=to_compression)
        result = read_stata(path, compression=read_compression, index_col='index')
        tm.assert_frame_equal(result, df)