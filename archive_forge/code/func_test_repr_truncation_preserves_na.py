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
def test_repr_truncation_preserves_na(self):
    df = DataFrame({'a': [pd.NA for _ in range(10)]})
    with option_context('display.max_rows', 2, 'display.show_dimensions', False):
        assert repr(df) == '       a\n0   <NA>\n..   ...\n9   <NA>'