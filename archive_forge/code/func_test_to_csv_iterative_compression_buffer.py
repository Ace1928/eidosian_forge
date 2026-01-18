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
def test_to_csv_iterative_compression_buffer(compression):
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD')), index=Index([f'i-{i}' for i in range(30)]))
    with io.BytesIO() as buffer:
        df.to_csv(buffer, compression=compression, chunksize=1)
        buffer.seek(0)
        tm.assert_frame_equal(pd.read_csv(buffer, compression=compression, index_col=0), df)
        assert not buffer.closed