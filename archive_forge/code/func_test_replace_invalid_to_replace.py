from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_invalid_to_replace(self):
    df = DataFrame({'one': ['a', 'b ', 'c'], 'two': ['d ', 'e ', 'f ']})
    msg = "Expecting 'to_replace' to be either a scalar, array-like, dict or None, got invalid type.*"
    msg2 = "DataFrame.replace without 'value' and with non-dict-like 'to_replace' is deprecated"
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            df.replace(lambda x: x.strip())