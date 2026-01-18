from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', ['1s', '-1s', '1us', '-1us', '1 day', '-1 day', '-23:59:59.999999', '-1 days +23:59:59.999999', '-1ns', '1ns', '-23:59:59.999999999'])
def test_td_from_repr_roundtrip(val):
    td = Timedelta(val)
    assert Timedelta(td._value) == td
    assert Timedelta(str(td)) == td
    assert Timedelta(td._repr_base(format='all')) == td
    assert Timedelta(td._repr_base()) == td