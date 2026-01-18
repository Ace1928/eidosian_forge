from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_pos_args_deprecation(self):
    df = DataFrame({'a': [1, 2, 3]})
    msg = "Starting with pandas version 3.0 all arguments of to_string except for the argument 'buf' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buf = StringIO()
        df.to_string(buf, None, None, True, True)