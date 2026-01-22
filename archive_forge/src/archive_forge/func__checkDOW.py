import re, time, datetime
from .utils import isStr
def _checkDOW(self):
    if self.dayOfWeek() > 4:
        raise NormalDateException("%r isn't a business day" % self.normalDate)