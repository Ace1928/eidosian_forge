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
class default_label_formatter(ParameterizedFunction):
    """Default formatter to turn parameter names into appropriate widget labels."""
    capitalize = Parameter(default=True, doc='\n        Whether or not the label should be capitalized.')
    replace_underscores = Parameter(default=True, doc='\n        Whether or not underscores should be replaced with spaces.')
    overrides = Parameter(default={}, doc='\n        Allows custom labels to be specified for specific parameter\n        names using a dictionary where key is the parameter name and the\n        value is the desired label.')

    def __call__(self, pname):
        if pname in self.overrides:
            return self.overrides[pname]
        if self.replace_underscores:
            pname = pname.replace('_', ' ')
        if self.capitalize:
            pname = pname[:1].upper() + pname[1:]
        return pname