from __future__ import annotations
import contextlib
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence as typing_Sequence
from typing import Tuple
from . import roles
from .base import _generative
from .base import Executable
from .base import SchemaVisitor
from .elements import ClauseElement
from .. import exc
from .. import util
from ..util import topological
from ..util.typing import Protocol
from ..util.typing import Self
def _can_create_table(self, table):
    self.dialect.validate_identifier(table.name)
    effective_schema = self.connection.schema_for_object(table)
    if effective_schema:
        self.dialect.validate_identifier(effective_schema)
    return not self.checkfirst or not self.dialect.has_table(self.connection, table.name, schema=effective_schema)