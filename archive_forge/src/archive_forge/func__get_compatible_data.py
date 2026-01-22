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
def _get_compatible_data(self, other):
    if isinstance(other, BigBitFieldData):
        data = other._buffer
    elif isinstance(other, (bytes, bytearray, memoryview)):
        data = other
    else:
        raise ValueError('Incompatible data-type')
    diff = len(data) - len(self)
    if diff > 0:
        self._buffer.extend(b'\x00' * diff)
    return data