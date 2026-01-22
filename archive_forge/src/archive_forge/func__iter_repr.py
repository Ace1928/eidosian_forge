import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
def _iter_repr(self, max_items=9):
    if len(self) <= max_items:
        for elt in conversion.noconversion(self):
            yield elt
    else:
        half_items = max_items // 2
        for i in range(0, half_items):
            yield self[i]
        yield '...'
        for i in range(-half_items, 0):
            yield self[i]