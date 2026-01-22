from __future__ import annotations
from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from .. import util
from ..util.typing import Literal
class SelectStatementRole(StatementRole, ReturnsRowsRole):
    __slots__ = ()
    _role_name = 'SELECT construct or equivalent text() construct'

    def subquery(self) -> Subquery:
        raise NotImplementedError('All SelectStatementRole objects should implement a .subquery() method.')