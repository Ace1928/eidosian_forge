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
def test_standard_colors_all(self):
    from matplotlib import colors
    from pandas.plotting._matplotlib.style import get_standard_colors
    for c in colors.cnames:
        result = get_standard_colors(num_colors=1, color=c)
        assert result == [c]
        result = get_standard_colors(num_colors=1, color=[c])
        assert result == [c]
        result = get_standard_colors(num_colors=3, color=c)
        assert result == [c] * 3
        result = get_standard_colors(num_colors=3, color=[c])
        assert result == [c] * 3
    for c in colors.ColorConverter.colors:
        result = get_standard_colors(num_colors=1, color=c)
        assert result == [c]
        result = get_standard_colors(num_colors=1, color=[c])
        assert result == [c]
        result = get_standard_colors(num_colors=3, color=c)
        assert result == [c] * 3
        result = get_standard_colors(num_colors=3, color=[c])
        assert result == [c] * 3