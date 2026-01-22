from datetime import datetime
import pytest
from pytz import utc
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
class TestHolidayCalendar(AbstractHolidayCalendar):
    rules = [USMartinLutherKingJr, holiday_1, holiday_2, USLaborDay]