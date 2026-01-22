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
def add_parameter(self_, param_name, param_obj):
    """
        Add a new Parameter object into this object's class.

        Should result in a Parameter equivalent to one declared
        in the class's source code.
        """
    cls = self_.cls
    type.__setattr__(cls, param_name, param_obj)
    ParameterizedMetaclass._initialize_parameter(cls, param_name, param_obj)
    cls._param__private.params.clear()