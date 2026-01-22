from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
from . import types
from .array import ARRAY
from ...sql import coercions
from ...sql import elements
from ...sql import expression
from ...sql import functions
from ...sql import roles
from ...sql import schema
from ...sql.schema import ColumnCollectionConstraint
from ...sql.sqltypes import TEXT
from ...sql.visitors import InternalTraversal
@property
def _from_objects(self):
    return self.target._from_objects + self.order_by._from_objects