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
class Greek(unicode_set):
    """Unicode set for Greek Unicode Character Ranges"""
    _ranges = [(880, 1023), (7936, 7957), (7960, 7965), (7968, 8005), (8008, 8013), (8016, 8023), (8025,), (8027,), (8029,), (8031, 8061), (8064, 8116), (8118, 8132), (8134, 8147), (8150, 8155), (8157, 8175), (8178, 8180), (8182, 8190)]