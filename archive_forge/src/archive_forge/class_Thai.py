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
class Thai(unicode_set):
    """Unicode set for Thai Unicode Character Range"""
    _ranges = [(3585, 3642), (3647, 3675)]