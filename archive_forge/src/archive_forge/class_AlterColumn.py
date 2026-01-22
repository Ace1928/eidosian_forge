from __future__ import annotations
import functools
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import exc
from sqlalchemy import Integer
from sqlalchemy import types as sqltypes
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import Column
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql.elements import quoted_name
from ..util.sqla_compat import _columns_for_constraint  # noqa
from ..util.sqla_compat import _find_columns  # noqa
from ..util.sqla_compat import _fk_spec  # noqa
from ..util.sqla_compat import _is_type_bound  # noqa
from ..util.sqla_compat import _table_for_constraint  # noqa
class AlterColumn(AlterTable):

    def __init__(self, name: str, column_name: str, schema: Optional[str]=None, existing_type: Optional[TypeEngine]=None, existing_nullable: Optional[bool]=None, existing_server_default: Optional[_ServerDefault]=None, existing_comment: Optional[str]=None) -> None:
        super().__init__(name, schema=schema)
        self.column_name = column_name
        self.existing_type = sqltypes.to_instance(existing_type) if existing_type is not None else None
        self.existing_nullable = existing_nullable
        self.existing_server_default = existing_server_default
        self.existing_comment = existing_comment