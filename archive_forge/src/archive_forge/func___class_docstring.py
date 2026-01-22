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
def __class_docstring(mcs):
    """
        Customize the class docstring with a Parameter table if
        `docstring_describe_params` and the `param_pager` is available.
        """
    if not docstring_describe_params or not param_pager:
        return
    class_docstr = mcs.__doc__ if mcs.__doc__ else ''
    description = param_pager(mcs)
    mcs.__doc__ = class_docstr + '\n' + description