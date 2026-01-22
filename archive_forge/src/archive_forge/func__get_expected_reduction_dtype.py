from typing import final
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype
def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
    return arr.dtype