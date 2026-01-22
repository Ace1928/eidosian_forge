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
def _find_pname(pclass):
    """
    Go up the stack and attempt to find a Parameter declaration of the form
    `pname = param.Parameter(` or `pname = pm.Parameter(`.
    """
    stack = traceback.extract_stack()
    for frame in stack:
        match = re.match('^(\\S+)\\s*=\\s*(param|pm)\\.' + pclass + '\\(', frame.line)
        if match:
            return match.group(1)