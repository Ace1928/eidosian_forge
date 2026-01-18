from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_read_excel_header_index_out_of_range(self, engine):
    with open('df_header_oob.xlsx', 'rb') as f:
        with pytest.raises(ValueError, match='exceeds maximum'):
            pd.read_excel(f, header=[0, 1])