from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_axis_name_raises(self):
    ser = Series([1])
    msg = 'No axis named 1 for object type Series'
    with pytest.raises(ValueError, match=msg):
        ser._set_axis_name(name='a', axis=1)