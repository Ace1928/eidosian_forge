from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class ConstraintColumnNotFoundError(ArgumentError):
    """raised when a constraint refers to a string column name that
    is not present in the table being constrained.

    .. versionadded:: 2.0

    """