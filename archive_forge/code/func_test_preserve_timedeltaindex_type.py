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
def test_preserve_timedeltaindex_type(setup_path):
    df = DataFrame(np.random.default_rng(2).normal(size=(10, 5)))
    df.index = timedelta_range(start='0s', periods=10, freq='1s', name='example')
    with ensure_clean_store(setup_path) as store:
        store['df'] = df
        tm.assert_frame_equal(store['df'], df)