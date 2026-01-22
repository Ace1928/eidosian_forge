import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _timegm(tt):
    year, month, mday, hour, min, sec = tt[:6]
    if year >= EPOCH_YEAR and 1 <= month <= 12 and (1 <= mday <= 31) and (0 <= hour <= 24) and (0 <= min <= 59) and (0 <= sec <= 61):
        return timegm(tt)
    else:
        return None