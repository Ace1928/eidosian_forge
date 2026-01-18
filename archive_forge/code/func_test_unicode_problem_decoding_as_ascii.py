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
def test_unicode_problem_decoding_as_ascii(self):
    df = DataFrame({'c/σ': Series({'test': np.nan})})
    str(df.to_string())