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
def test_pickle_path_localpath():
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    result = tm.round_trip_pathlib(lambda p: df.to_hdf(p, key='df'), lambda p: read_hdf(p, 'df'))
    tm.assert_frame_equal(df, result)