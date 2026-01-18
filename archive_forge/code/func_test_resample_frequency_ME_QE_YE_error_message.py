from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
@pytest.mark.parametrize('freq,freq_depr', [('2M', '2ME'), ('2Q', '2QE'), ('2Q-FEB', '2QE-FEB'), ('2Y', '2YE'), ('2Y-MAR', '2YE-MAR'), ('2M', '2me'), ('2Q', '2qe'), ('2Y-MAR', '2ye-mar')])
def test_resample_frequency_ME_QE_YE_error_message(series_and_frame, freq, freq_depr):
    msg = f"for Period, please use '{freq[1:]}' instead of '{freq_depr[1:]}'"
    obj = series_and_frame
    with pytest.raises(ValueError, match=msg):
        obj.resample(freq_depr)