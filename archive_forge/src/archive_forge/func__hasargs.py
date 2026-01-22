import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _hasargs(type_, *args):
    try:
        res = all((arg in type_.__args__ for arg in args))
    except AttributeError:
        return False
    except TypeError:
        if type_.__args__ is None:
            return False
        else:
            raise
    else:
        return res