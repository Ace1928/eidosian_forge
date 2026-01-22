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
def __bind_database__(meth):

    @wraps(meth)
    def inner(self, *args, **kwargs):
        result = meth(self, *args, **kwargs)
        if self._database:
            return result.bind(self._database)
        return result
    return inner