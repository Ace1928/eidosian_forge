import sys
import datetime
import locale as _locale
from itertools import repeat
class IllegalMonthError(ValueError):

    def __init__(self, month):
        self.month = month

    def __str__(self):
        return 'bad month number %r; must be 1-12' % self.month