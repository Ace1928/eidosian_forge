import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def _check_cast(df, v):
    """
    Check if all dtypes of df are equal to v
    """
    assert all((s.dtype.name == v for _, s in df.items()))