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
class CalendarDateRange(Range):
    """
    A date range specified as (start_date, end_date).
    """

    def _validate_value(self, val, allow_None):
        if allow_None and val is None:
            return
        for n in val:
            if not isinstance(n, dt.date):
                raise ValueError(f'{_validate_error_prefix(self)} only takes date types, not {val}.')
        start, end = val
        if not end >= start:
            raise ValueError(f'{_validate_error_prefix(self)} end date {val[1]} is before start date {val[0]}.')

    def _validate_bound_type(self, value, position, kind):
        if not isinstance(value, dt.date):
            raise ValueError(f'{_validate_error_prefix(self)} {position} {kind} can only be None or a date value, not {type(value)}.')

    @classmethod
    def serialize(cls, value):
        if value is None:
            return None
        return [v.strftime('%Y-%m-%d') for v in value]

    @classmethod
    def deserialize(cls, value):
        if value == 'null' or value is None:
            return None
        return tuple([dt.datetime.strptime(v, '%Y-%m-%d').date() for v in value])