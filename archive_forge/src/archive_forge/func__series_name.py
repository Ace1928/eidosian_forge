from datetime import datetime
import numpy as np
import pytest
from pandas import (
@pytest.fixture
def _series_name():
    """
    Fixture for parametrization of Series name for Series used with
    date_range, period_range and timedelta_range indexes
    """
    return None