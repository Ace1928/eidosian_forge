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
def test_tz_utc():
    dt = datetime.datetime(1970, 1, 1, tzinfo=mdates.UTC)
    assert dt.tzname() == 'UTC'