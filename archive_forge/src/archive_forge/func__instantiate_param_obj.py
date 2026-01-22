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
def _instantiate_param_obj(paramobj, owner=None):
    """Return a Parameter object suitable for instantiation given the class's Parameter object"""
    p = copy.copy(paramobj)
    p.owner = owner
    p.watchers = {}
    for s in p.__class__._all_slots_:
        v = getattr(p, s)
        if _is_mutable_container(v) and s != 'default':
            setattr(p, s, copy.copy(v))
    return p