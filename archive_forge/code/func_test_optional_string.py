from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_optional_string():
    out = string_like('apple', 'value')
    assert out == 'apple'
    out = string_like('apple', 'value', options=('apple', 'banana', 'cherry'))
    assert out == 'apple'
    out = string_like(None, 'value', optional=True)
    assert out is None
    out = string_like(None, 'value', optional=True, options=('apple', 'banana', 'cherry'))
    assert out is None
    with pytest.raises(TypeError, match='value must be a string'):
        string_like(1, 'value', optional=True)
    with pytest.raises(TypeError, match='value must be a string'):
        string_like(b'4', 'value', optional=True)