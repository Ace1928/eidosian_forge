import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_take_pandas_style_negative_raises(self, data, na_value):
    with pytest.raises(ValueError, match=''):
        data.take([0, -2], fill_value=na_value, allow_fill=True)