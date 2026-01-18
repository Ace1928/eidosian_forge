from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_raise_error_on_datetime_time_data(self):
    df = DataFrame(np.random.default_rng(2).standard_normal(10), columns=['a'])
    df['dtime'] = date_range(start='2014-01-01', freq='h', periods=10).time
    msg = "must be a string or a (real )?number, not 'datetime.time'"
    with pytest.raises(TypeError, match=msg):
        df.plot(kind='scatter', x='dtime', y='a')