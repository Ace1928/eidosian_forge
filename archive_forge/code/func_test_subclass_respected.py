from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_subclass_respected():

    class MyCustomTimedelta(Timedelta):
        pass
    td = MyCustomTimedelta('1 minute')
    assert isinstance(td, MyCustomTimedelta)