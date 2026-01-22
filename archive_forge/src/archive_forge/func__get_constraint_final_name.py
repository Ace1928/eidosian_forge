from __future__ import annotations
import contextlib
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import __version__
from sqlalchemy import inspect
from sqlalchemy import schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from sqlalchemy.engine import url
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.schema import Column
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.sql import visitors
from sqlalchemy.sql.base import DialectKWArgs
from sqlalchemy.sql.elements import BindParameter
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.elements import quoted_name
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.elements import UnaryExpression
from sqlalchemy.sql.visitors import traverse
from typing_extensions import TypeGuard
def _get_constraint_final_name(constraint: Union[Index, Constraint], dialect: Optional[Dialect]) -> Optional[str]:
    if constraint.name is None:
        return None
    assert dialect is not None
    if sqla_14:
        return dialect.identifier_preparer.format_constraint(constraint, _alembic_quote=False)
    else:
        if hasattr(constraint.name, 'quote'):
            quoted_name_cls: type = type(constraint.name)
        else:
            quoted_name_cls = quoted_name
        new_name = quoted_name_cls(str(constraint.name), quote=False)
        constraint = constraint.__class__(name=new_name)
        if isinstance(constraint, schema.Index):
            d = dialect.ddl_compiler(dialect, None)
            return d._prepared_index_name(constraint)
        else:
            return dialect.identifier_preparer.format_constraint(constraint)