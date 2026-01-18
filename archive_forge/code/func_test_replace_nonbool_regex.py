import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('frame', [False, True])
def test_replace_nonbool_regex(self, frame):
    obj = pd.Series(['a', 'b', 'c '])
    if frame:
        obj = obj.to_frame()
    msg = "'to_replace' must be 'None' if 'regex' is not a bool"
    with pytest.raises(ValueError, match=msg):
        obj.replace(to_replace=['a'], regex='foo')