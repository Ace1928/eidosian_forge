from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_use_bottleneck():
    if nanops._BOTTLENECK_INSTALLED:
        with pd.option_context('use_bottleneck', True):
            assert pd.get_option('use_bottleneck')
        with pd.option_context('use_bottleneck', False):
            assert not pd.get_option('use_bottleneck')