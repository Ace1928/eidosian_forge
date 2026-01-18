import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_tab_completion_cat_str(self):
    s = Series(date_range('1/1/2015', periods=5)).astype('category')
    assert 'cat' in dir(s)
    assert 'str' not in dir(s)
    assert 'dt' in dir(s)