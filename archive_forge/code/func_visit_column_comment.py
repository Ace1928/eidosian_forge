from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from sqlalchemy.sql import sqltypes
from .base import AddColumn
from .base import alter_table
from .base import ColumnComment
from .base import ColumnDefault
from .base import ColumnName
from .base import ColumnNullable
from .base import ColumnType
from .base import format_column_name
from .base import format_server_default
from .base import format_table_name
from .base import format_type
from .base import IdentityColumnDefault
from .base import RenameTable
from .impl import DefaultImpl
from ..util.sqla_compat import compiles
@compiles(ColumnComment, 'oracle')
def visit_column_comment(element: ColumnComment, compiler: OracleDDLCompiler, **kw) -> str:
    ddl = 'COMMENT ON COLUMN {table_name}.{column_name} IS {comment}'
    comment = compiler.sql_compiler.render_literal_value(element.comment if element.comment is not None else '', sqltypes.String())
    return ddl.format(table_name=element.table_name, column_name=element.column_name, comment=comment)