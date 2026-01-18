import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categorical_empty(self):
    msg = 'No Categoricals to union'
    with pytest.raises(ValueError, match=msg):
        union_categoricals([])