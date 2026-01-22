from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar
from . import types
from .array import ARRAY
from ...sql import coercions
from ...sql import elements
from ...sql import expression
from ...sql import functions
from ...sql import roles
from ...sql import schema
from ...sql.schema import ColumnCollectionConstraint
from ...sql.sqltypes import TEXT
from ...sql.visitors import InternalTraversal
class ts_headline(_regconfig_fn):
    """The PostgreSQL ``ts_headline`` SQL function.

    This function applies automatic casting of the REGCONFIG argument
    to use the :class:`_postgresql.REGCONFIG` datatype automatically,
    and applies a return type of :class:`_types.TEXT`.

    Assuming the PostgreSQL dialect has been imported, either by invoking
    ``from sqlalchemy.dialects import postgresql``, or by creating a PostgreSQL
    engine using ``create_engine("postgresql...")``,
    :class:`_postgresql.ts_headline` will be used automatically when invoking
    ``sqlalchemy.func.ts_headline()``, ensuring the correct argument and return
    type handlers are used at compile and execution time.

    .. versionadded:: 2.0.0rc1

    """
    inherit_cache = True
    type = TEXT

    def __init__(self, *args, **kwargs):
        args = list(args)
        if len(args) < 2:
            has_regconfig = False
        elif isinstance(args[1], elements.ColumnElement) and args[1].type._type_affinity is types.TSQUERY:
            has_regconfig = False
        else:
            has_regconfig = True
        if has_regconfig:
            initial_arg = coercions.expect(roles.ExpressionElementRole, args.pop(0), apply_propagate_attrs=self, name=getattr(self, 'name', None), type_=types.REGCONFIG)
            initial_arg = [initial_arg]
        else:
            initial_arg = []
        addtl_args = [coercions.expect(roles.ExpressionElementRole, c, name=getattr(self, 'name', None), apply_propagate_attrs=self) for c in args]
        super().__init__(*initial_arg + addtl_args, **kwargs)