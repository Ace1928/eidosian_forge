from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['B', 'C'])
def test_not_equals_bday(self, freq):
    rng = date_range('2009-01-01', '2010-01-01', freq=freq)
    assert not rng.equals(list(rng))