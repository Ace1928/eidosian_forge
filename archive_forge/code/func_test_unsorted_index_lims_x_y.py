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
def test_unsorted_index_lims_x_y(self):
    df = DataFrame({'y': [0.0, 1.0, 2.0, 3.0], 'z': [91.0, 90.0, 93.0, 92.0]})
    ax = df.plot(x='z', y='y')
    xmin, xmax = ax.get_xlim()
    lines = ax.get_lines()
    assert xmin <= np.nanmin(lines[0].get_data()[0])
    assert xmax >= np.nanmax(lines[0].get_data()[0])