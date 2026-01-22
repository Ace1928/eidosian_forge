from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
def _timestamp_date_part(date_part):

    def dec(self):
        db = self.model._meta.database
        expr = self / Value(self.resolution, converter=False) if self.resolution > 1 else self
        return db.extract_date(date_part, db.from_timestamp(expr))
    return dec