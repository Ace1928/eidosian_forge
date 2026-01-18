import datetime
import re
@staticmethod
def yearly(t):
    y = t.year + 1
    return t.replace(year=y, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)