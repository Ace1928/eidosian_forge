from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
def test_transform_none_to_type():
    df = DataFrame({'a': [None]})
    msg = 'argument must be a'
    with pytest.raises(TypeError, match=msg):
        df.transform({'a': lambda x: int(x.iloc[0])})