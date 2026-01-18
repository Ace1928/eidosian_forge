from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_timestamp_raises(self, index, frame_or_series):
    obj = frame_or_series(index=index, dtype=object)
    if not isinstance(index, PeriodIndex):
        msg = f'unsupported Type {type(index).__name__}'
        with pytest.raises(TypeError, match=msg):
            obj.to_timestamp()