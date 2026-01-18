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
def test_truncate_with_different_dtypes_multiindex(self):
    df = DataFrame({'Vals': range(100)})
    frame = pd.concat([df], keys=['Sweep'], names=['Sweep', 'Index'])
    result = repr(frame)
    result2 = repr(frame.iloc[:5])
    assert result.startswith(result2)