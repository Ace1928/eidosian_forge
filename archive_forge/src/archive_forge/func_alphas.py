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
@_lazyclassproperty
def alphas(cls):
    """all alphabetic characters in this range"""
    return u''.join(filter(unicode.isalpha, cls._get_chars_for_ranges()))