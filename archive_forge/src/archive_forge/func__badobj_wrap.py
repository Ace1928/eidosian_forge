from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def _badobj_wrap(self, value, func, allow_complex=True, **kwargs):
    if value.dtype.kind == 'O':
        if allow_complex:
            value = value.astype('c16')
        else:
            value = value.astype('f8')
    return func(value, **kwargs)