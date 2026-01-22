import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
class DateRange(Range):
    """
    A datetime or date range specified as (start, end).

    Bounds must be specified as datetime or date types (see param.dt_types).
    """

    def _validate_bound_type(self, value, position, kind):
        if not isinstance(value, dt_types):
            raise ValueError(f'{_validate_error_prefix(self)} {position} {kind} can only be None or a date/datetime value, not {type(value)}.')

    def _validate_bounds(self, val, bounds, inclusive_bounds, kind):
        val = None if val is None else tuple(map(_to_datetime, val))
        bounds = None if bounds is None else tuple(map(_to_datetime, bounds))
        super()._validate_bounds(val, bounds, inclusive_bounds, kind)

    def _validate_value(self, val, allow_None):
        if allow_None and val is None:
            return
        if not isinstance(val, tuple):
            raise ValueError(f'{_validate_error_prefix(self)} only takes a tuple value, not {type(val)}.')
        for n in val:
            if isinstance(n, dt_types):
                continue
            raise ValueError(f'{_validate_error_prefix(self)} only takes date/datetime values, not {type(n)}.')
        start, end = val
        if not end >= start:
            raise ValueError(f'{_validate_error_prefix(self)} end datetime {val[1]} is before start datetime {val[0]}.')

    @classmethod
    def serialize(cls, value):
        if value is None:
            return None
        serialized = []
        for v in value:
            if not isinstance(v, (dt.datetime, dt.date)):
                v = v.astype(dt.datetime)
            if type(v) == dt.date:
                v = v.strftime('%Y-%m-%d')
            else:
                v = v.strftime('%Y-%m-%dT%H:%M:%S.%f')
            serialized.append(v)
        return serialized

    def deserialize(cls, value):
        if value == 'null' or value is None:
            return None
        deserialized = []
        for v in value:
            if len(v) == 10:
                v = dt.datetime.strptime(v, '%Y-%m-%d').date()
            else:
                v = dt.datetime.strptime(v, '%Y-%m-%dT%H:%M:%S.%f')
            deserialized.append(v)
        return tuple(deserialized)