import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _get_type_origin(type_):
    """Some spaghetti logic to accommodate differences between 3.6 and 3.7 in
    the typing api"""
    try:
        origin = type_.__origin__
    except AttributeError:
        origin = _NO_TYPE_ORIGIN
    if sys.version_info.minor == 6:
        try:
            origin = type_.__extra__
        except AttributeError:
            origin = type_
        else:
            origin = type_ if origin in (None, _NO_TYPE_ORIGIN) else origin
    elif origin is _NO_TYPE_ORIGIN:
        origin = type_
    return origin