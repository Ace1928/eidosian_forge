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
def _null_ordering_case(self, nulls):
    if nulls.lower() == 'last':
        ifnull, notnull = (1, 0)
    elif nulls.lower() == 'first':
        ifnull, notnull = (0, 1)
    else:
        raise ValueError('unsupported value for nulls= ordering.')
    return Case(None, ((self.node.is_null(), ifnull),), notnull)