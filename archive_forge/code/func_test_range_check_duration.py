import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_range_check_duration(self):
    with self.assertRaises(YearOutOfBoundsError):
        PythonTimeBuilder.build_duration(PnY=str(datetime.timedelta.max.days // 365 + 1))
    with self.assertRaises(MonthOutOfBoundsError):
        PythonTimeBuilder.build_duration(PnM=str(datetime.timedelta.max.days // 30 + 1))
    with self.assertRaises(DayOutOfBoundsError):
        PythonTimeBuilder.build_duration(PnD=str(datetime.timedelta.max.days + 1))
    with self.assertRaises(WeekOutOfBoundsError):
        PythonTimeBuilder.build_duration(PnW=str(datetime.timedelta.max.days // 7 + 1))
    with self.assertRaises(HoursOutOfBoundsError):
        PythonTimeBuilder.build_duration(TnH=str(datetime.timedelta.max.days * 24 + 24))
    with self.assertRaises(MinutesOutOfBoundsError):
        PythonTimeBuilder.build_duration(TnM=str(datetime.timedelta.max.days * 24 * 60 + 24 * 60))
    with self.assertRaises(SecondsOutOfBoundsError):
        PythonTimeBuilder.build_duration(TnS=str(datetime.timedelta.max.days * 24 * 60 * 60 + 24 * 60 * 60))
    maxpart = datetime.timedelta.max.days // 7
    with self.assertRaises(DayOutOfBoundsError):
        PythonTimeBuilder.build_duration(PnY=str(maxpart // 365 + 1), PnM=str(maxpart // 30 + 1), PnD=str(maxpart + 1), PnW=str(maxpart // 7 + 1), TnH=str(maxpart * 24 + 1), TnM=str(maxpart * 24 * 60 + 1), TnS=str(maxpart * 24 * 60 * 60 + 1))