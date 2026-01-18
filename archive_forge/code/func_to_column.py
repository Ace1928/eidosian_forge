from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
def to_column(self, migration_context: Optional[MigrationContext]=None) -> Column[Any]:
    if self._reverse is not None:
        return self._reverse.column
    schema_obj = schemaobj.SchemaObjects(migration_context)
    return schema_obj.column(self.column_name, NULLTYPE)