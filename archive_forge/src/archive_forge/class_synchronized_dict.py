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
class synchronized_dict(dict):

    def __init__(self, *args, **kwargs):
        super(synchronized_dict, self).__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def __getitem__(self, key):
        with self._lock:
            return super(synchronized_dict, self).__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            return super(synchronized_dict, self).__setitem__(key, value)

    def __delitem__(self, key):
        with self._lock:
            return super(synchronized_dict, self).__delitem__(key)