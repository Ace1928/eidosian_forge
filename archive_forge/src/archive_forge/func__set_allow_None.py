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
def _set_allow_None(self, allow_None):
    if self.default is None:
        self.allow_None = True
    elif allow_None is not Undefined:
        self.allow_None = allow_None
    else:
        self.allow_None = self._slot_defaults['allow_None']