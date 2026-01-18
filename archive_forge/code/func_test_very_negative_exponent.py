from io import StringIO
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('neg_exp', [-617, -100000, pytest.param(-99999999999999999, marks=pytest.mark.skip_ubsan)])
def test_very_negative_exponent(all_parsers_all_precisions, neg_exp):
    parser, precision = all_parsers_all_precisions
    data = f'data\n10E{neg_exp}'
    result = parser.read_csv(StringIO(data), float_precision=precision)
    expected = DataFrame({'data': [0.0]})
    tm.assert_frame_equal(result, expected)