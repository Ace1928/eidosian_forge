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
def _render_identity(identity: Identity, autogen_context: AutogenContext) -> str:
    kwargs = sqla_compat._get_identity_options_dict(identity, dialect_kwargs=True)
    return '%(prefix)sIdentity(%(kwargs)s)' % {'prefix': _sqlalchemy_autogenerate_prefix(autogen_context), 'kwargs': ', '.join(('%s=%s' % pair for pair in kwargs.items()))}