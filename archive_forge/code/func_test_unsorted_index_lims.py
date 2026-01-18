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
@pytest.mark.parametrize('df', [DataFrame({'y': [0.0, 1.0, 2.0, 3.0]}, index=[1.0, 0.0, 3.0, 2.0]), DataFrame({'y': [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0]}, index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0])])
def test_unsorted_index_lims(self, df):
    ax = df.plot()
    xmin, xmax = ax.get_xlim()
    lines = ax.get_lines()
    assert xmin <= np.nanmin(lines[0].get_data()[0])
    assert xmax >= np.nanmax(lines[0].get_data()[0])