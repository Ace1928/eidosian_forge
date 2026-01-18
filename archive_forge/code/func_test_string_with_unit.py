from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('constructor, value, unit, expectation', [(Timedelta, '10s', 'ms', (ValueError, 'unit must not be specified')), (to_timedelta, '10s', 'ms', (ValueError, 'unit must not be specified')), (to_timedelta, ['1', 2, 3], 's', (ValueError, 'unit must not be specified'))])
def test_string_with_unit(constructor, value, unit, expectation):
    exp, match = expectation
    with pytest.raises(exp, match=match):
        _ = constructor(value, unit=unit)