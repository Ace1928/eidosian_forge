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
def test_parse_datetime_string_with_reso_check_instance_type_raise_exception():
    msg = "Argument 'date_string' has incorrect type (expected str, got tuple)"
    with pytest.raises(TypeError, match=re.escape(msg)):
        parse_datetime_string_with_reso((1, 2, 3))
    result = parse_datetime_string_with_reso('2019')
    expected = (datetime(2019, 1, 1), 'year')
    assert result == expected