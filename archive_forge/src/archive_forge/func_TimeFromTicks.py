import datetime
import time
import collections.abc
from _sqlite3 import *
def TimeFromTicks(ticks):
    return Time(*time.localtime(ticks)[3:6])