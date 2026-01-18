import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('strrep', ['243.164', '245.968', '249.585', '259.745', '265.742', '272.567', '279.196', '280.366', '275.034', '271.351', '272.889', '270.627', '280.828', '290.383', '308.153', '319.945', '336.0', '344.09', '351.385', '356.178', '359.82', '361.03', '367.701', '380.812', '387.98', '391.749', '391.171', '385.97', '385.345', '386.121', '390.996', '399.734', '413.073', '421.532', '430.221', '437.092', '439.746', '446.01', '451.191', '460.463', '469.779', '472.025', '479.49', '474.864', '467.54', '471.978'])
def test_precision_float_conversion(strrep):
    result = to_numeric(strrep)
    assert result == float(strrep)