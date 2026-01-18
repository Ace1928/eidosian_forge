from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rfloordiv_intarray(self):
    ints = np.array([1349654400, 1349740800, 1349827200, 1349913600]) * 10 ** 9
    msg = 'Invalid dtype'
    with pytest.raises(TypeError, match=msg):
        ints // Timedelta(1, unit='s')