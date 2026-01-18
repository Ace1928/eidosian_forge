from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def to_interval(interval: str | Literal) -> Interval:
    """Builds an interval expression from a string like '1 day' or '5 months'."""
    if isinstance(interval, Literal):
        if not interval.is_string:
            raise ValueError('Invalid interval string.')
        interval = interval.this
    interval_parts = INTERVAL_STRING_RE.match(interval)
    if not interval_parts:
        raise ValueError('Invalid interval string.')
    return Interval(this=Literal.string(interval_parts.group(1)), unit=Var(this=interval_parts.group(2).upper()))