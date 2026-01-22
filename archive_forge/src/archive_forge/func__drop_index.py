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
@renderers.dispatch_for(ops.DropIndexOp)
def _drop_index(autogen_context: AutogenContext, op: ops.DropIndexOp) -> str:
    index = op.to_index()
    has_batch = autogen_context._has_batch
    if has_batch:
        tmpl = '%(prefix)sdrop_index(%(name)r%(kwargs)s)'
    else:
        tmpl = '%(prefix)sdrop_index(%(name)r, table_name=%(table_name)r%(schema)s%(kwargs)s)'
    opts = _render_dialect_kwargs_items(autogen_context, index)
    text = tmpl % {'prefix': _alembic_autogenerate_prefix(autogen_context), 'name': _render_gen_name(autogen_context, op.index_name), 'table_name': _ident(op.table_name), 'schema': ', schema=%r' % _ident(op.schema) if op.schema else '', 'kwargs': ', ' + ', '.join(opts) if opts else ''}
    return text