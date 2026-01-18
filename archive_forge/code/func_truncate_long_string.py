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
def truncate_long_string(data: str, max_length: Optional[int]=None) -> str:
    """Truncate arguments with representation longer than max_length

    Args:
        data (str): The data to truncate
        max_length (Optional[int], optional): The max length. Defaults to None.

    Returns:
        truncated (str): The truncated string
    """
    if max_length is None:
        return data
    return data[:max_length] + '...' if len(data) > max_length else data