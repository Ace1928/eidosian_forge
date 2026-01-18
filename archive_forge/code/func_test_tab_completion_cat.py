import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_tab_completion_cat(self):
    s = Series(list('abbcd'), dtype='category')
    assert 'cat' in dir(s)
    assert 'str' in dir(s)
    assert 'dt' not in dir(s)