import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
def accept_arguments(f):
    """
    Decorator for decorators that accept arguments
    """

    @functools.wraps(f)
    def _f(*args, **kwargs):
        return lambda actual_f: f(actual_f, *args, **kwargs)
    return _f