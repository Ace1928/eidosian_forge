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
def watch_values(self_, fn, parameter_names, what='value', onlychanged=True, queued=False, precedence=0):
    """
        Easier-to-use version of `watch` specific to watching for changes in parameter values.

        Only allows `what` to be 'value', and invokes the callback `fn` using keyword
        arguments <param_name>=<new_value> rather than with a list of Event objects.
        """
    if precedence < 0:
        raise ValueError('User-defined watch callbacks must declare a positive precedence. Negative precedences are reserved for internal Watchers.')
    assert what == 'value'
    if isinstance(parameter_names, list):
        parameter_names = tuple(parameter_names)
    else:
        parameter_names = (parameter_names,)
    watcher = Watcher(inst=self_.self, cls=self_.cls, fn=fn, mode='kwargs', onlychanged=onlychanged, parameter_names=parameter_names, what=what, queued=queued, precedence=precedence)
    self_._register_watcher('append', watcher, what)
    return watcher