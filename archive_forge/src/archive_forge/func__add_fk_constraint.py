from __future__ import annotations
from io import StringIO
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from mako.pygen import PythonPrinter
from sqlalchemy import schema as sa_schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from sqlalchemy.sql.elements import conv
from sqlalchemy.sql.elements import quoted_name
from .. import util
from ..operations import ops
from ..util import sqla_compat
@renderers.dispatch_for(ops.CreateForeignKeyOp)
def _add_fk_constraint(autogen_context: AutogenContext, op: ops.CreateForeignKeyOp) -> str:
    args = [repr(_render_gen_name(autogen_context, op.constraint_name))]
    if not autogen_context._has_batch:
        args.append(repr(_ident(op.source_table)))
    args.extend([repr(_ident(op.referent_table)), repr([_ident(col) for col in op.local_cols]), repr([_ident(col) for col in op.remote_cols])])
    kwargs = ['referent_schema', 'onupdate', 'ondelete', 'initially', 'deferrable', 'use_alter', 'match']
    if not autogen_context._has_batch:
        kwargs.insert(0, 'source_schema')
    for k in kwargs:
        if k in op.kw:
            value = op.kw[k]
            if value is not None:
                args.append('%s=%r' % (k, value))
    return '%(prefix)screate_foreign_key(%(args)s)' % {'prefix': _alembic_autogenerate_prefix(autogen_context), 'args': ', '.join(args)}