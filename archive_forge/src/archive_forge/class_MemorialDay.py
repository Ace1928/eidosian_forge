from datetime import datetime
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.tseries.holiday import (
class MemorialDay(AbstractHolidayCalendar):
    rules = [USMemorialDay]