from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_repr_truncation_column_size(self):
    df = DataFrame({'a': [108480, 30830], 'b': [12345, 12345], 'c': [12345, 12345], 'd': [12345, 12345], 'e': ['a' * 50] * 2})
    assert '...' in str(df)
    assert '    ...    ' not in str(df)