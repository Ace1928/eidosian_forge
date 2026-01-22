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
def _escape_like_expr(self, s, template):
    if s.find('_') >= 0 or s.find('%') >= 0 or s.find('\\') >= 0:
        s = s.replace('\\', '\\\\').replace('_', '\\_').replace('%', '\\%')
        return NodeList((Value(template % s, converter=False), SQL('ESCAPE'), Value('\\', converter=False)))
    return template % s