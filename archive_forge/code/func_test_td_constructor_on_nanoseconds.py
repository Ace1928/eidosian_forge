from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('constructed_td, conversion', [(Timedelta(nanoseconds=100), '100ns'), (Timedelta(days=1, hours=1, minutes=1, weeks=1, seconds=1, milliseconds=1, microseconds=1, nanoseconds=1), 694861001001001), (Timedelta(microseconds=1) + Timedelta(nanoseconds=1), '1us1ns'), (Timedelta(microseconds=1) - Timedelta(nanoseconds=1), '999ns'), (Timedelta(microseconds=1) + 5 * Timedelta(nanoseconds=-2), '990ns')])
def test_td_constructor_on_nanoseconds(constructed_td, conversion):
    assert constructed_td == Timedelta(conversion)