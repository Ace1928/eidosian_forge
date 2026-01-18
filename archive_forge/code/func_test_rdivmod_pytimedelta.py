from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_rdivmod_pytimedelta(self):
    result = divmod(timedelta(days=2, hours=6), Timedelta(days=1))
    assert result[0] == 2
    assert isinstance(result[1], Timedelta)
    assert result[1] == Timedelta(hours=6)