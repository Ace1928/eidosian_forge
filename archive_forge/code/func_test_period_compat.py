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
def test_period_compat(self):
    df = DataFrame(np.random.default_rng(2).random((21, 2)), index=bdate_range(datetime(2000, 1, 1), datetime(2000, 1, 31)), columns=['a', 'b'])
    df.plot()
    mpl.pyplot.axhline(y=0)