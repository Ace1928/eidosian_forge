from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('arg', [[1, 2], 1])
def test_to_timedelta_invalid_unit(self, arg):
    msg = 'invalid unit abbreviation: foo'
    with pytest.raises(ValueError, match=msg):
        to_timedelta(arg, unit='foo')