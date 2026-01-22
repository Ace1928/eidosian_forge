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
def _create_year_locator(date1, date2, **kwargs):
    locator = mdates.YearLocator(**kwargs)
    locator.create_dummy_axis()
    locator.axis.set_view_interval(mdates.date2num(date1), mdates.date2num(date2))
    return locator