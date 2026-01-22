import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _get_type_cons(type_):
    """More spaghetti logic for 3.6 vs. 3.7"""
    if sys.version_info.minor == 6:
        try:
            cons = type_.__extra__
        except AttributeError:
            try:
                cons = type_.__origin__
            except AttributeError:
                cons = type_
            else:
                cons = type_ if cons is None else cons
        else:
            try:
                cons = type_.__origin__ if cons is None else cons
            except AttributeError:
                cons = type_
    else:
        cons = type_.__origin__
    return cons