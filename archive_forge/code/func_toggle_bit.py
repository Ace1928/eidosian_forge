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
def toggle_bit(self, idx):
    byte_num, byte_offset = self._ensure_length(idx)
    self._buffer[byte_num] ^= 1 << byte_offset
    return bool(self._buffer[byte_num] & 1 << byte_offset)