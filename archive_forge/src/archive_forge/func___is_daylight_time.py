from suds import UnicodeMixin
import datetime
import re
import time
def __is_daylight_time(self, dt):
    if not time.daylight:
        return False
    time_tuple = dt.replace(tzinfo=None).timetuple()
    time_tuple = time.localtime(time.mktime(time_tuple))
    return time_tuple.tm_isdst > 0