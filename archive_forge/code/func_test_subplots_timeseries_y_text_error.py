import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_subplots_timeseries_y_text_error(self):
    data = {'numeric': np.array([1, 2, 5]), 'text': ['This', 'should', 'fail']}
    testdata = DataFrame(data)
    msg = 'no numeric data to plot'
    with pytest.raises(TypeError, match=msg):
        testdata.plot(y='text')