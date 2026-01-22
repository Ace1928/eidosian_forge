from calendar import timegm
from datetime import datetime, time, timedelta
from email.utils import parsedate_tz, mktime_tz
import re
import aniso8601
import pytz
def _expand_datetime(start, value):
    if not isinstance(start, datetime):
        end = start + timedelta(days=1)
    else:
        time = value.split('T')[1]
        time_without_offset = re.sub('[+-].+', '', time)
        num_separators = time_without_offset.count(':')
        if num_separators == 0:
            end = start + timedelta(hours=1)
        elif num_separators == 1:
            end = start + timedelta(minutes=1)
        else:
            end = start + timedelta(seconds=1)
    return end