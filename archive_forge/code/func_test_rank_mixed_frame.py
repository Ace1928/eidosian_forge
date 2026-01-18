from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
def test_rank_mixed_frame(self, float_string_frame):
    float_string_frame['datetime'] = datetime.now()
    float_string_frame['timedelta'] = timedelta(days=1, seconds=1)
    float_string_frame.rank(numeric_only=False)
    with pytest.raises(TypeError, match='not supported between instances of'):
        float_string_frame.rank(axis=1)