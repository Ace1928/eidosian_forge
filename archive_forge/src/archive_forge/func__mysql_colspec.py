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
def _mysql_colspec(compiler: MySQLDDLCompiler, nullable: Optional[bool], server_default: Optional[Union[_ServerDefault, Literal[False]]], type_: TypeEngine, autoincrement: Optional[bool], comment: Optional[Union[str, Literal[False]]]) -> str:
    spec = '%s %s' % (compiler.dialect.type_compiler.process(type_), 'NULL' if nullable else 'NOT NULL')
    if autoincrement:
        spec += ' AUTO_INCREMENT'
    if server_default is not False and server_default is not None:
        spec += ' DEFAULT %s' % format_server_default(compiler, server_default)
    if comment:
        spec += ' COMMENT %s' % compiler.sql_compiler.render_literal_value(comment, sqltypes.String())
    return spec