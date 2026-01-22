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
def _m_caller(self, method_name, what='value', changed=None, callback=None):
    """
    Wraps a method call adding support for scheduling a callback
    before it is executed and skipping events if a subobject has
    changed but its values have not.
    """
    function = getattr(self, method_name)
    _caller = _async_caller if iscoroutinefunction(function) else _sync_caller
    caller = partial(_caller, what=what, changed=changed, callback=callback, function=function)
    caller._watcher_name = method_name
    return caller