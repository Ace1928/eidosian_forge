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
class DuplicateColumnError(ArgumentError):
    """a Column is being added to a Table that would replace another
    Column, without appropriate parameters to allow this in place.

    .. versionadded:: 2.0.0b4

    """