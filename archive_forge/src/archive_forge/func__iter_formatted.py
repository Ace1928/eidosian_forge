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
def _iter_formatted(self, max_items=9):
    ln = len(self)
    half_items = max_items // 2
    if ln == 0:
        return
    elif ln < max_items:
        str_vec = StrVector(as_character(self))
    else:
        str_vec = r_concat(as_character(self.rx(IntSexpVector(tuple(range(1, half_items - 1))))), StrSexpVector(['...']), as_character(self.rx(IntSexpVector(tuple(range(ln - half_items, ln))))))
    for str_elt in str_vec:
        yield self.repr_format_elt(str_elt)