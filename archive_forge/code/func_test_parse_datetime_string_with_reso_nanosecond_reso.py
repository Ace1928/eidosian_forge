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
def test_parse_datetime_string_with_reso_nanosecond_reso():
    parsed, reso = parse_datetime_string_with_reso('2022-04-20 09:19:19.123456789')
    assert reso == 'nanosecond'