from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import schema
from sqlalchemy import types as sqltypes
from .base import alter_table
from .base import AlterColumn
from .base import ColumnDefault
from .base import ColumnName
from .base import ColumnNullable
from .base import ColumnType
from .base import format_column_name
from .base import format_server_default
from .impl import DefaultImpl
from .. import util
from ..util import sqla_compat
from ..util.sqla_compat import _is_mariadb
from ..util.sqla_compat import _is_type_bound
from ..util.sqla_compat import compiles
@compiles(MySQLAlterDefault, 'mysql', 'mariadb')
def _mysql_alter_default(element: MySQLAlterDefault, compiler: MySQLDDLCompiler, **kw) -> str:
    return '%s ALTER COLUMN %s %s' % (alter_table(compiler, element.table_name, element.schema), format_column_name(compiler, element.column_name), 'SET DEFAULT %s' % format_server_default(compiler, element.default) if element.default is not None else 'DROP DEFAULT')