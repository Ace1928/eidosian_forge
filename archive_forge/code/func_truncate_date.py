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
def truncate_date(self, date_part, date_field):
    return fn.DATE_FORMAT(date_field, __mysql_date_trunc__[date_part], python_value=simple_date_time)