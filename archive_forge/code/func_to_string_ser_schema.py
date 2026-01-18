from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def to_string_ser_schema(*, when_used: WhenUsed='json-unless-none') -> ToStringSerSchema:
    """
    Returns a schema for serialization using python's `str()` / `__str__` method.

    Args:
        when_used: Same meaning as for [general_function_plain_ser_schema], but with a different default
    """
    s = dict(type='to-string')
    if when_used != 'json-unless-none':
        s['when_used'] = when_used
    return s