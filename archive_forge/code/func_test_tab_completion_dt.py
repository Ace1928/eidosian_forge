import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_tab_completion_dt(self):
    s = Series(date_range('1/1/2015', periods=5))
    assert 'dt' in dir(s)
    assert 'str' not in dir(s)
    assert 'cat' not in dir(s)