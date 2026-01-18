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
@pytest.mark.slow
def test_series_grid_settings(self):
    pytest.importorskip('scipy')
    _check_grid_settings(Series([1, 2, 3]), plotting.PlotAccessor._series_kinds + plotting.PlotAccessor._common_kinds)