import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def _check_ax_limits(col, ax):
    y_min, y_max = ax.get_ylim()
    assert y_min <= col.min()
    assert y_max >= col.max()