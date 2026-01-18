import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def try_modin_df_almost_equals_compare(df1, df2):
    """Compare two dataframes as nearly equal if possible, otherwise compare as completely equal."""
    dtypes1, dtypes2 = [dtype if is_list_like((dtype := df.dtypes)) else [dtype] for df in (df1, df2)]
    if all(map(is_numeric_dtype, dtypes1)) and all(map(is_numeric_dtype, dtypes2)):
        modin_df_almost_equals_pandas(df1, df2)
    else:
        df_equals(df1, df2)