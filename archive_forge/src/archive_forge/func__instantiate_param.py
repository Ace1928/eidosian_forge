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
def _instantiate_param(self_, param_obj, dict_=None, key=None, deepcopy=True):
    instantiator = copy.deepcopy if deepcopy else lambda o: o
    self = self_.self
    dict_ = dict_ or self._param__private.values
    key = key or param_obj.name
    if shared_parameters._share:
        param_key = (str(type(self)), param_obj.name)
        if param_key in shared_parameters._shared_cache:
            new_object = shared_parameters._shared_cache[param_key]
        else:
            new_object = instantiator(param_obj.default)
            shared_parameters._shared_cache[param_key] = new_object
    else:
        new_object = instantiator(param_obj.default)
    dict_[key] = new_object
    if isinstance(new_object, Parameterized):
        global object_count
        object_count += 1
        new_object.param._generate_name()