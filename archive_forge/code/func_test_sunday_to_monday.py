from datetime import datetime
import pytest
from pandas.tseries.holiday import (
def test_sunday_to_monday():
    assert sunday_to_monday(_SUNDAY) == _MONDAY