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
class _ParametersRestorer:
    """
    Context-manager to handle the reset of parameter values after an update.
    """

    def __init__(self, *, parameters, restore, refs=None):
        self._parameters = parameters
        self._restore = restore
        self._refs = {} if refs is None else refs

    def __enter__(self):
        return self._restore

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            self._parameters._update(dict(self._restore, **self._refs))
        finally:
            self._restore = {}