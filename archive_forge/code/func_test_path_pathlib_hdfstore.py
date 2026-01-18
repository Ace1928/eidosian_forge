import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_path_pathlib_hdfstore():
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))

    def writer(path):
        with HDFStore(path) as store:
            df.to_hdf(store, key='df')

    def reader(path):
        with HDFStore(path) as store:
            return read_hdf(store, 'df')
    result = tm.round_trip_pathlib(writer, reader)
    tm.assert_frame_equal(df, result)