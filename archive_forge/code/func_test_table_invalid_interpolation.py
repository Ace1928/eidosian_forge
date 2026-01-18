import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_table_invalid_interpolation(self):
    with pytest.raises(ValueError, match='Invalid interpolation: foo'):
        DataFrame(range(1)).quantile(0.5, method='table', interpolation='foo')