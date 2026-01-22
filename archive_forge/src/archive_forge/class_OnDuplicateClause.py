from __future__ import annotations
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
from ... import exc
from ... import util
from ...sql._typing import _DMLTableArgument
from ...sql.base import _exclusive_against
from ...sql.base import _generative
from ...sql.base import ColumnCollection
from ...sql.base import ReadOnlyColumnCollection
from ...sql.dml import Insert as StandardInsert
from ...sql.elements import ClauseElement
from ...sql.elements import KeyedColumnElement
from ...sql.expression import alias
from ...sql.selectable import NamedFromClause
from ...util.typing import Self
class OnDuplicateClause(ClauseElement):
    __visit_name__ = 'on_duplicate_key_update'
    _parameter_ordering: Optional[List[str]] = None
    stringify_dialect = 'mysql'

    def __init__(self, inserted_alias: NamedFromClause, update: _UpdateArg) -> None:
        self.inserted_alias = inserted_alias
        if isinstance(update, list) and (update and isinstance(update[0], tuple)):
            self._parameter_ordering = [key for key, value in update]
            update = dict(update)
        if isinstance(update, dict):
            if not update:
                raise ValueError('update parameter dictionary must not be empty')
        elif isinstance(update, ColumnCollection):
            update = dict(update)
        else:
            raise ValueError('update parameter must be a non-empty dictionary or a ColumnCollection such as the `.c.` collection of a Table object')
        self.update = update