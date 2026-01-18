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
def unregister_table_function(self, name):
    for idx, klass in enumerate(self._table_functions):
        if klass.name == name:
            break
    else:
        return False
    self._table_functions.pop(idx)
    return True