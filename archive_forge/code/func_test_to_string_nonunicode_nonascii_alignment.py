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
def test_to_string_nonunicode_nonascii_alignment(self):
    df = DataFrame([['aaÃ¤Ã¤', 1], ['bbbb', 2]])
    rep_str = df.to_string()
    lines = rep_str.split('\n')
    assert len(lines[1]) == len(lines[2])