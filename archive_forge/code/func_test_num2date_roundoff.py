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
def test_num2date_roundoff():
    assert mdates.num2date(100000.0000578702) == datetime.datetime(2243, 10, 17, 0, 0, 4, 999980, tzinfo=datetime.timezone.utc)
    assert mdates.num2date(100000.0000578703) == datetime.datetime(2243, 10, 17, 0, 0, 5, tzinfo=datetime.timezone.utc)