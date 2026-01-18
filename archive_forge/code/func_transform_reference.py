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
def transform_reference(arg):
    """
    Applies transforms to turn objects which should be treated like
    a parameter reference into a valid reference that can be resolved
    by Param. This is useful for adding handling for depending on objects
    that are not simple Parameters or functions with dependency
    definitions.
    """
    for transform in _reference_transforms:
        if isinstance(arg, Parameter) or hasattr(arg, '_dinfo'):
            break
        arg = transform(arg)
    return arg