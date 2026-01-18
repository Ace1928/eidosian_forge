import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_requires_integer_n(self, obj):
    with pytest.raises(ValueError, match='Only integers accepted as `n` values'):
        obj.sample(n=3.2)