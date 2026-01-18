import numpy as np
import pytest
import pandas as pd
from pandas import Index
@pytest.fixture(params=[pd.offsets.Hour(2), pd.offsets.Minute(120), pd.Timedelta(hours=2).to_pytimedelta(), pd.Timedelta(seconds=2 * 3600), np.timedelta64(2, 'h'), np.timedelta64(120, 'm')], ids=lambda x: type(x).__name__)
def two_hours(request):
    """
    Several timedelta-like and DateOffset objects that each represent
    a 2-hour timedelta
    """
    return request.param