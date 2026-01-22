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
def _compare_stacked_y_cood(self, normal_lines, stacked_lines):
    base = np.zeros(len(normal_lines[0].get_data()[1]))
    for nl, sl in zip(normal_lines, stacked_lines):
        base += nl.get_data()[1]
        sy = sl.get_data()[1]
        tm.assert_numpy_array_equal(base, sy)