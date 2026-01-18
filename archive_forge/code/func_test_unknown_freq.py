from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_unknown_freq():
    with pytest.raises(ValueError, match='freq is not understood by pandas'):
        CalendarTimeTrend('unknown', True, order=3)