from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
@pytest.mark.parametrize('time,format_expected', [(0, '00:00'), (86399.999999, '23:59:59.999999'), (90000, '01:00'), (3723, '01:02:03'), (39723.2, '11:02:03.200')])
def test_time_formatter(self, time, format_expected):
    result = converter.TimeFormatter(None)(time)
    assert result == format_expected