import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@pytest.mark.parametrize('np_scalar', [True, False])
def test_sequence_duration(np_scalar):
    td1 = datetime.timedelta(2, 3601, 1)
    td2 = datetime.timedelta(1, 100, 1000)
    if np_scalar:
        data = [np.timedelta64(td1), None, np.timedelta64(td2)]
    else:
        data = [td1, None, td2]
    arr = pa.array(data)
    assert len(arr) == 3
    assert arr.type == pa.duration('us')
    assert arr.null_count == 1
    assert arr[0].as_py() == td1
    assert arr[1].as_py() is None
    assert arr[2].as_py() == td2