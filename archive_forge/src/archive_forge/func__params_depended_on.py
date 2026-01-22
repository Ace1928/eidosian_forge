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
def _params_depended_on(minfo, dynamic=True, intermediate=True):
    """
    Resolves dependencies declared on a Parameterized method.
    Dynamic dependencies, i.e. dependencies on sub-objects which may
    or may not yet be available, are only resolved if dynamic=True.
    By default intermediate dependencies, i.e. dependencies on the
    path to a sub-object are returned. For example for a dependency
    on 'a.b.c' dependencies on 'a' and 'b' are returned as long as
    intermediate=True.

    Returns lists of concrete dependencies on available parameters
    and dynamic dependencies specifications which have to resolved
    if the referenced sub-objects are defined.
    """
    deps, dynamic_deps = ([], [])
    dinfo = getattr(minfo.method, '_dinfo', {})
    for d in dinfo.get('dependencies', list(minfo.cls.param)):
        ddeps, ddynamic_deps = (minfo.cls if minfo.inst is None else minfo.inst).param._spec_to_obj(d, dynamic, intermediate)
        dynamic_deps += ddynamic_deps
        for dep in ddeps:
            if isinstance(dep, PInfo):
                deps.append(dep)
            else:
                method_deps, method_dynamic_deps = _params_depended_on(dep, dynamic, intermediate)
                deps += method_deps
                dynamic_deps += method_dynamic_deps
    return (deps, dynamic_deps)