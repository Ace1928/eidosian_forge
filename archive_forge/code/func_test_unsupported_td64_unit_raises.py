import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['Y', 'M'])
def test_unsupported_td64_unit_raises(unit):
    with pytest.raises(ValueError, match=f"Unit {unit} is not supported. Only unambiguous timedelta values durations are supported. Allowed units are 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'"):
        Timedelta(np.timedelta64(1, unit))