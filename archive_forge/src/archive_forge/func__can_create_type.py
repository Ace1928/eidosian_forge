from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from ... import schema
from ... import util
from ...sql import coercions
from ...sql import elements
from ...sql import roles
from ...sql import sqltypes
from ...sql import type_api
from ...sql.base import _NoArg
from ...sql.ddl import InvokeCreateDDLBase
from ...sql.ddl import InvokeDropDDLBase
def _can_create_type(self, type_):
    if not self.checkfirst:
        return True
    effective_schema = self.connection.schema_for_object(type_)
    return not self.connection.dialect.has_type(self.connection, type_.name, schema=effective_schema)