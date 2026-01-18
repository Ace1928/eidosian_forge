import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_unicode_index(setup_path):
    unicode_values = ['σ', 'σσ']
    s = Series(np.random.default_rng(2).standard_normal(len(unicode_values)), unicode_values)
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)