import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.skipif(not np_version_gt2, reason='StringDType only available in numpy 2 and above')
@pytest.mark.parametrize('data', [{'a': ['a', 'b', 'c'], 'b': [1.0, 2.0, 3.0], 'c': ['d', 'e', 'f']}])
def test_np_string_array_object_cast(self, data):
    from numpy.dtypes import StringDType
    data['a'] = np.array(data['a'], dtype=StringDType())
    res = DataFrame(data)
    assert res['a'].dtype == np.object_
    assert (res['a'] == data['a']).all()