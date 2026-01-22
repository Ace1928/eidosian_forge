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
def __get_signature(mcs):
    """
        For classes with a constructor signature that matches the default
        Parameterized.__init__ signature (i.e. ``__init__(self, **params)``)
        this method will generate a new signature that expands the
        parameters. If the signature differs from the default the
        custom signature is returned.
        """
    if mcs._param__private.signature:
        return mcs._param__private.signature
    if inspect.signature(mcs.__init__) != DEFAULT_SIGNATURE:
        return None
    processed_kws, keyword_groups = (set(), [])
    for cls in reversed(mcs.mro()):
        keyword_group = []
        for k, v in sorted(cls.__dict__.items()):
            if isinstance(v, Parameter) and k not in processed_kws and (not v.readonly):
                keyword_group.append(k)
                processed_kws.add(k)
        keyword_groups.append(keyword_group)
    keywords = [el for grp in reversed(keyword_groups) for el in grp]
    mcs._param__private.signature = signature = inspect.Signature([inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY) for k in keywords])
    return signature