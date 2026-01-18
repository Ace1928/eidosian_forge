from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import CDay
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
from pandas.tseries.holiday import USFederalHolidayCalendar

Tests for offsets.CustomBusinessDay / CDay
