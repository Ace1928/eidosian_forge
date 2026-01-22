from __future__ import annotations
from typing import Any
from .._typing import _OnConflictIndexElementsT
from .._typing import _OnConflictIndexWhereT
from .._typing import _OnConflictSetT
from .._typing import _OnConflictWhereT
from ... import util
from ...sql import coercions
from ...sql import roles
from ...sql._typing import _DMLTableArgument
from ...sql.base import _exclusive_against
from ...sql.base import _generative
from ...sql.base import ColumnCollection
from ...sql.base import ReadOnlyColumnCollection
from ...sql.dml import Insert as StandardInsert
from ...sql.elements import ClauseElement
from ...sql.elements import KeyedColumnElement
from ...sql.expression import alias
from ...util.typing import Self
class OnConflictClause(ClauseElement):
    stringify_dialect = 'sqlite'
    constraint_target: None
    inferred_target_elements: _OnConflictIndexElementsT
    inferred_target_whereclause: _OnConflictIndexWhereT

    def __init__(self, index_elements: _OnConflictIndexElementsT=None, index_where: _OnConflictIndexWhereT=None):
        if index_elements is not None:
            self.constraint_target = None
            self.inferred_target_elements = index_elements
            self.inferred_target_whereclause = index_where
        else:
            self.constraint_target = self.inferred_target_elements = self.inferred_target_whereclause = None