import datetime
import time
import collections.abc
from _sqlite3 import *
def TimestampFromTicks(ticks):
    return Timestamp(*time.localtime(ticks)[:6])