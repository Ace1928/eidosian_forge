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
def test_to_string_unicode_columns(self, float_frame):
    df = DataFrame({'σ': np.arange(10.0)})
    buf = StringIO()
    df.to_string(buf=buf)
    buf.getvalue()
    buf = StringIO()
    df.info(buf=buf)
    buf.getvalue()
    result = float_frame.to_string()
    assert isinstance(result, str)