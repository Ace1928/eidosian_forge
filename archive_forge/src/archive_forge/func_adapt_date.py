import datetime
import time
import collections.abc
from _sqlite3 import *
def adapt_date(val):
    return val.isoformat()