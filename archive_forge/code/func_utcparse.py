import calendar
import datetime
import datetime as dt
import importlib
import logging
import numbers
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from redis.exceptions import ResponseError
from .exceptions import TimeoutFormatError
def utcparse(string: str) -> dt.datetime:
    try:
        return datetime.datetime.strptime(string, _TIMESTAMP_FORMAT)
    except ValueError:
        return datetime.datetime.strptime(string, '%Y-%m-%dT%H:%M:%SZ')