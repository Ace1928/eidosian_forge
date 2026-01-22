import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _get_type_args(tp: Type, default: Union[Tuple[Type, ...], _NoArgs]=_NO_ARGS) -> Union[Tuple[Type, ...], _NoArgs]:
    if hasattr(tp, '__args__'):
        if tp.__args__ is not None:
            return tp.__args__
    return default