from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def while_changing(expression: Expression, func: t.Callable[[Expression], E]) -> E:
    """
    Applies a transformation to a given expression until a fix point is reached.

    Args:
        expression: The expression to be transformed.
        func: The transformation to be applied.

    Returns:
        The transformed expression.
    """
    while True:
        for n in reversed(tuple(expression.walk())):
            n._hash = hash(n)
        start = hash(expression)
        expression = func(expression)
        for n in expression.walk():
            n._hash = None
        if start == hash(expression):
            break
    return expression