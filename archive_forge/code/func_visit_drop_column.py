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
@compiles(DropColumn)
def visit_drop_column(element: DropColumn, compiler: DDLCompiler, **kw) -> str:
    return '%s %s' % (alter_table(compiler, element.table_name, element.schema), drop_column(compiler, element.column.name, **kw))