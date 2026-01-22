import datetime
import time
import collections.abc
from _sqlite3 import *
def DateFromTicks(ticks):
    return Date(*time.localtime(ticks)[:3])