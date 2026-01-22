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
def _render_computed(computed: Computed, autogen_context: AutogenContext) -> str:
    text = _render_potential_expr(computed.sqltext, autogen_context, wrap_in_text=False)
    kwargs = {}
    if computed.persisted is not None:
        kwargs['persisted'] = computed.persisted
    return '%(prefix)sComputed(%(text)s, %(kwargs)s)' % {'prefix': _sqlalchemy_autogenerate_prefix(autogen_context), 'text': text, 'kwargs': ', '.join(('%s=%s' % pair for pair in kwargs.items()))}