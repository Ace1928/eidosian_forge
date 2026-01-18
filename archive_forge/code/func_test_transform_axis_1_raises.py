from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
def test_transform_axis_1_raises():
    msg = 'No axis named 1 for object type Series'
    with pytest.raises(ValueError, match=msg):
        Series([1]).transform('sum', axis=1)