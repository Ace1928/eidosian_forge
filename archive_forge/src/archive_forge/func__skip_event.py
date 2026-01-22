import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
def _skip_event(*events, **kwargs):
    """
    Checks whether a subobject event should be skipped.
    Returns True if all the values on the new subobject
    match the values on the previous subobject.
    """
    what = kwargs.get('what', 'value')
    changed = kwargs.get('changed')
    if changed is None:
        return False
    for e in events:
        for p in changed:
            if what == 'value':
                old = Undefined if e.old is None else _getattrr(e.old, p, None)
                new = Undefined if e.new is None else _getattrr(e.new, p, None)
            else:
                old = Undefined if e.old is None else _getattrr(e.old.param[p], what, None)
                new = Undefined if e.new is None else _getattrr(e.new.param[p], what, None)
            if not Comparator.is_equal(old, new):
                return False
    return True