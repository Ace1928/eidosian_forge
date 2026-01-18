from __future__ import annotations
import collections
import datetime as dt
import functools
import inspect
import json
import re
import typing
import warnings
from collections.abc import Mapping
from email.utils import format_datetime, parsedate_to_datetime
from pprint import pprint as py_pprint
from marshmallow.base import FieldABC
from marshmallow.exceptions import FieldInstanceResolutionError
from marshmallow.warnings import RemovedInMarshmallow4Warning
def timedelta_to_microseconds(value: dt.timedelta) -> int:
    """Compute the total microseconds of a timedelta

    https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Lib/datetime.py#L665-L667  # noqa: B950
    """
    return (value.days * (24 * 3600) + value.seconds) * 1000000 + value.microseconds