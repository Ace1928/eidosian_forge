from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_bad_value(self):
    msg = "Could not convert 'foo' to NumPy timedelta"
    with pytest.raises(ValueError, match=msg):
        to_timedelta(['foo', 'bar'])