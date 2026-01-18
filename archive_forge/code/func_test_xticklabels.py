from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_xticklabels(self):
    s = Series(np.arange(10), index=[f'P{i:02d}' for i in range(10)])
    _, ax = mpl.pyplot.subplots()
    ax = s.plot(xticks=[0, 3, 5, 9], ax=ax)
    exp = [f'P{i:02d}' for i in [0, 3, 5, 9]]
    _check_text_labels(ax.get_xticklabels(), exp)