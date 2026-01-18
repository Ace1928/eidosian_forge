from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('method_name', ['center', 'ljust', 'rjust', 'zfill', 'pad'])
def test_pad_width_bad_arg_raises(method_name, any_string_dtype):
    s = Series(['1', '22', 'a', 'bb'], dtype=any_string_dtype)
    op = operator.methodcaller(method_name, 'f')
    msg = 'width must be of integer type, not str'
    with pytest.raises(TypeError, match=msg):
        op(s.str)