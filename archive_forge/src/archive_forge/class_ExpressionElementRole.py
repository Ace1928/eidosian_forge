from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class ExpressionElementRole(TypedColumnsClauseRole[_T_co]):
    __slots__ = ()
    _role_name = 'SQL expression element'

    def label(self, name: Optional[str]) -> Label[_T]:
        raise NotImplementedError()