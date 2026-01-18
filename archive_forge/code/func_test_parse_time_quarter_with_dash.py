from datetime import datetime
import re
from dateutil.parser import parse as du_parse
from dateutil.tz import tzlocal
from hypothesis import given
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas._testing as tm
from pandas._testing._hypothesis import DATETIME_NO_TZ
@pytest.mark.parametrize('dashed,normal', [('1988-Q2', '1988Q2'), ('2Q-1988', '2Q1988')])
def test_parse_time_quarter_with_dash(dashed, normal):
    parsed_dash, reso_dash = parse_datetime_string_with_reso(dashed)
    parsed, reso = parse_datetime_string_with_reso(normal)
    assert parsed_dash == parsed
    assert reso_dash == reso