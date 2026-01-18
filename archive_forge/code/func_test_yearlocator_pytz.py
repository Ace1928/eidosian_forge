import datetime
import dateutil.tz
import dateutil.rrule
import functools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import rc_context, style
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.ticker as mticker
@pytest.mark.pytz
def test_yearlocator_pytz():
    pytz = pytest.importorskip('pytz')
    tz = pytz.timezone('America/New_York')
    x = [tz.localize(datetime.datetime(2010, 1, 1)) + datetime.timedelta(i) for i in range(2000)]
    locator = mdates.AutoDateLocator(interval_multiples=True, tz=tz)
    locator.create_dummy_axis()
    locator.axis.set_view_interval(mdates.date2num(x[0]) - 1.0, mdates.date2num(x[-1]) + 1.0)
    t = np.array([733408.208333, 733773.208333, 734138.208333, 734503.208333, 734869.208333, 735234.208333, 735599.208333])
    t = t + mdates.date2num(np.datetime64('0000-12-31'))
    np.testing.assert_allclose(t, locator())
    expected = ['2009-01-01 00:00:00-05:00', '2010-01-01 00:00:00-05:00', '2011-01-01 00:00:00-05:00', '2012-01-01 00:00:00-05:00', '2013-01-01 00:00:00-05:00', '2014-01-01 00:00:00-05:00', '2015-01-01 00:00:00-05:00']
    st = list(map(str, mdates.num2date(locator(), tz=tz)))
    assert st == expected
    assert np.allclose(locator.tick_values(x[0], x[1]), np.array([14610.20833333, 14610.33333333, 14610.45833333, 14610.58333333, 14610.70833333, 14610.83333333, 14610.95833333, 14611.08333333, 14611.20833333]))
    assert np.allclose(locator.get_locator(x[1], x[0]).tick_values(x[0], x[1]), np.array([14610.20833333, 14610.33333333, 14610.45833333, 14610.58333333, 14610.70833333, 14610.83333333, 14610.95833333, 14611.08333333, 14611.20833333]))