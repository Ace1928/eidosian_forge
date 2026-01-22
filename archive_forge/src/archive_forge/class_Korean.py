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
class Korean(unicode_set):
    """Unicode set for Korean Unicode Character Range"""
    _ranges = [(44032, 55215), (4352, 4607), (12592, 12687), (43360, 43391), (55216, 55295), (12288, 12351)]