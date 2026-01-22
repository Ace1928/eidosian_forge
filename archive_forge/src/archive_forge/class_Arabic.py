import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
class Arabic(unicode_set):
    """Unicode set for Arabic Unicode Character Range"""
    _ranges = [(1536, 1563), (1566, 1791), (1792, 1919)]