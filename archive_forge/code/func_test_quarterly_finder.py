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
@pytest.mark.parametrize('year_span', [11.25, 30, 80, 150, 400, 800, 1500, 2500, 3500])
def test_quarterly_finder(year_span):
    vmin = -1000
    vmax = vmin + year_span * 4
    span = vmax - vmin + 1
    if span < 45:
        pytest.skip('the quarterly finder is only invoked if the span is >= 45')
    nyears = span / 4
    min_anndef, maj_anndef = converter._get_default_annual_spacing(nyears)
    result = converter._quarterly_finder(vmin, vmax, to_offset('QE'))
    quarters = PeriodIndex(arrays.PeriodArray(np.array([x[0] for x in result]), dtype='period[Q]'))
    majors = np.array([x[1] for x in result])
    minors = np.array([x[2] for x in result])
    major_quarters = quarters[majors]
    minor_quarters = quarters[minors]
    check_major_years = major_quarters.year % maj_anndef == 0
    check_minor_years = minor_quarters.year % min_anndef == 0
    check_major_quarters = major_quarters.quarter == 1
    check_minor_quarters = minor_quarters.quarter == 1
    assert np.all(check_major_years)
    assert np.all(check_minor_years)
    assert np.all(check_major_quarters)
    assert np.all(check_minor_quarters)