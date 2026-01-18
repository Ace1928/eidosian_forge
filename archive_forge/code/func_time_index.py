from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.fixture(scope='module')
def time_index(request):
    return pd.date_range('2000-01-01', periods=833, freq='B')