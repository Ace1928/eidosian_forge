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
@pytest.mark.parametrize('dtype', [None, 'category'])
def test_to_hdf_with_object_column_names_should_run(tmp_path, setup_path, dtype):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=Index(['a', 'b'], dtype=dtype))
    path = tmp_path / setup_path
    df.to_hdf(path, key='df', format='table', data_columns=True)
    result = read_hdf(path, 'df', where=f'index = [{df.index[0]}]')
    assert len(result)