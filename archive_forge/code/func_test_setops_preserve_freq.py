from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('tz', [None, 'Asia/Tokyo', 'US/Eastern'])
def test_setops_preserve_freq(self, tz):
    rng = date_range('1/1/2000', '1/1/2002', name='idx', tz=tz)
    result = rng[:50].union(rng[50:100])
    assert result.name == rng.name
    assert result.freq == rng.freq
    assert result.tz == rng.tz
    result = rng[:50].union(rng[30:100])
    assert result.name == rng.name
    assert result.freq == rng.freq
    assert result.tz == rng.tz
    result = rng[:50].union(rng[60:100])
    assert result.name == rng.name
    assert result.freq is None
    assert result.tz == rng.tz
    result = rng[:50].intersection(rng[25:75])
    assert result.name == rng.name
    assert result.freqstr == 'D'
    assert result.tz == rng.tz
    nofreq = DatetimeIndex(list(rng[25:75]), name='other')
    result = rng[:50].union(nofreq)
    assert result.name is None
    assert result.freq == rng.freq
    assert result.tz == rng.tz
    result = rng[:50].intersection(nofreq)
    assert result.name is None
    assert result.freq == rng.freq
    assert result.tz == rng.tz