from __future__ import annotations
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import types as sqltypes
from sqlalchemy.schema import Column
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.base import Executable
from sqlalchemy.sql.elements import ClauseElement
from .base import AddColumn
from .base import alter_column
from .base import alter_table
from .base import ColumnDefault
from .base import ColumnName
from .base import ColumnNullable
from .base import ColumnType
from .base import format_column_name
from .base import format_server_default
from .base import format_table_name
from .base import format_type
from .base import RenameTable
from .impl import DefaultImpl
from .. import util
from ..util import sqla_compat
from ..util.sqla_compat import compiles
@compiles(_ExecDropConstraint, 'mssql')
def _exec_drop_col_constraint(element: _ExecDropConstraint, compiler: MSSQLCompiler, **kw) -> str:
    schema, tname, colname, type_ = (element.schema, element.tname, element.colname, element.type_)
    return "declare @const_name varchar(256)\nselect @const_name = QUOTENAME([name]) from %(type)s\nwhere parent_object_id = object_id('%(schema_dot)s%(tname)s')\nand col_name(parent_object_id, parent_column_id) = '%(colname)s'\nexec('alter table %(tname_quoted)s drop constraint ' + @const_name)" % {'type': type_, 'tname': tname, 'colname': colname, 'tname_quoted': format_table_name(compiler, tname, schema), 'schema_dot': schema + '.' if schema else ''}