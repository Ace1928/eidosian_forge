import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _determine_attrib_eq_order(cmp, eq, order, default_eq):
    """
    Validate the combination of *cmp*, *eq*, and *order*. Derive the effective
    values of eq and order.  If *eq* is None, set it to *default_eq*.
    """
    if cmp is not None and any((eq is not None, order is not None)):
        msg = "Don't mix `cmp` with `eq' and `order`."
        raise ValueError(msg)

    def decide_callable_or_boolean(value):
        """
        Decide whether a key function is used.
        """
        if callable(value):
            value, key = (True, value)
        else:
            key = None
        return (value, key)
    if cmp is not None:
        cmp, cmp_key = decide_callable_or_boolean(cmp)
        return (cmp, cmp_key, cmp, cmp_key)
    if eq is None:
        eq, eq_key = (default_eq, None)
    else:
        eq, eq_key = decide_callable_or_boolean(eq)
    if order is None:
        order, order_key = (eq, eq_key)
    else:
        order, order_key = decide_callable_or_boolean(order)
    if eq is False and order is True:
        msg = '`order` can only be True if `eq` is True too.'
        raise ValueError(msg)
    return (eq, eq_key, order, order_key)