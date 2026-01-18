import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def utc_time_sans_frac():
    return int('%d' % time.mktime(time.gmtime()))