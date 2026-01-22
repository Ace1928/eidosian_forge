import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
def _reset_epoch_test_example():
    """
    Reset the Matplotlib date epoch so it can be set again.

    Only for use in tests and examples.
    """
    global _epoch
    _epoch = None