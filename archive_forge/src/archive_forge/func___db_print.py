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
def __db_print(self_, level, msg, *args, **kw):
    """
        Calls the logger returned by the get_logger() function,
        prepending the result of calling dbprint_prefix() (if any).

        See python's logging module for details.
        """
    self_or_cls = self_.self_or_cls
    if get_logger(name=self_or_cls.name).isEnabledFor(level):
        if dbprint_prefix and callable(dbprint_prefix):
            msg = dbprint_prefix() + ': ' + msg
        get_logger(name=self_or_cls.name).log(level, msg, *args, **kw)