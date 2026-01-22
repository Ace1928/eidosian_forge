from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class StrAsPlainColumnRole(ColumnArgumentRole):
    __slots__ = ()
    _role_name = 'Column expression or string key'