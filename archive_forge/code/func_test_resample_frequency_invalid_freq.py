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
@pytest.mark.parametrize('freq_depr', ['2BME', '2CBME', '2SME', '2BQE-FEB', '2BYE-MAR'])
def test_resample_frequency_invalid_freq(series_and_frame, freq_depr):
    msg = f'Invalid frequency: {freq_depr[1:]}'
    obj = series_and_frame
    with pytest.raises(ValueError, match=msg):
        obj.resample(freq_depr)