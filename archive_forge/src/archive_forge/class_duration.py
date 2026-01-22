import datetime
import hashlib
import heapq
import math
import os
import random
import re
import sys
import threading
import zlib
from peewee import format_date_time
@aggregate(DATE)
class duration(object):

    def __init__(self):
        self._min = self._max = None

    def step(self, value):
        dt = format_date_time_sqlite(value)
        if self._min is None or dt < self._min:
            self._min = dt
        if self._max is None or dt > self._max:
            self._max = dt

    def finalize(self):
        if self._min and self._max:
            td = self._max - self._min
            return total_seconds(td)
        return None