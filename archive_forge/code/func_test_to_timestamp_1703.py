from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_timestamp_1703(self):
    index = period_range('1/1/2012', periods=4, freq='D')
    result = index.to_timestamp()
    assert result[0] == Timestamp('1/1/2012')