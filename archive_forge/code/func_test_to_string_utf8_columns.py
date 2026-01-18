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
def test_to_string_utf8_columns(self):
    n = 'א'.encode()
    df = DataFrame([1, 2], columns=[n])
    with option_context('display.max_rows', 1):
        repr(df)