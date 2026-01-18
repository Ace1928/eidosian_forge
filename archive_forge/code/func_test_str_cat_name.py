import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.mark.parametrize('other', [None, Series, Index])
def test_str_cat_name(index_or_series, other):
    box = index_or_series
    values = ['a', 'b']
    if other:
        other = other(values)
    else:
        other = values
    result = box(values, name='name').str.cat(other, sep=',')
    assert result.name == 'name'