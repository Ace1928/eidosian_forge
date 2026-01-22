from __future__ import annotations
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import Numeric
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import BIGINT
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import NULLTYPE
from .base import alter_column
from .base import alter_table
from .base import AlterColumn
from .base import ColumnComment
from .base import format_column_name
from .base import format_table_name
from .base import format_type
from .base import IdentityColumnDefault
from .base import RenameTable
from .impl import ComparisonResult
from .impl import DefaultImpl
from .. import util
from ..autogenerate import render
from ..operations import ops
from ..operations import schemaobj
from ..operations.base import BatchOperations
from ..operations.base import Operations
from ..util import sqla_compat
from ..util.sqla_compat import compiles
def _render_potential_column(value: Union[ColumnClause[Any], Column[Any], TextClause, FunctionElement[Any]], autogen_context: AutogenContext) -> str:
    if isinstance(value, ColumnClause):
        if value.is_literal:
            template = '%(prefix)sliteral_column(%(name)r)'
        else:
            template = '%(prefix)scolumn(%(name)r)'
        return template % {'prefix': render._sqlalchemy_autogenerate_prefix(autogen_context), 'name': value.name}
    else:
        return render._render_potential_expr(value, autogen_context, wrap_in_text=isinstance(value, (TextClause, FunctionElement)))