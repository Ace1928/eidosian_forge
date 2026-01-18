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
@pytest.mark.parametrize('format', ['fixed', 'table'])
def test_to_hdf_errors(tmp_path, format, setup_path):
    data = ['\ud800foo']
    ser = Series(data, index=Index(data))
    path = tmp_path / setup_path
    ser.to_hdf(path, key='table', format=format, errors='surrogatepass')
    result = read_hdf(path, 'table', errors='surrogatepass')
    tm.assert_series_equal(result, ser)