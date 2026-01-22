from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from functools import wraps
import re
from . import dictionary
from .types import _OracleBoolean
from .types import _OracleDate
from .types import BFILE
from .types import BINARY_DOUBLE
from .types import BINARY_FLOAT
from .types import DATE
from .types import FLOAT
from .types import INTERVAL
from .types import LONG
from .types import NCLOB
from .types import NUMBER
from .types import NVARCHAR2  # noqa
from .types import OracleRaw  # noqa
from .types import RAW
from .types import ROWID  # noqa
from .types import TIMESTAMP
from .types import VARCHAR2  # noqa
from ... import Computed
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import default
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import and_
from ...sql import bindparam
from ...sql import compiler
from ...sql import expression
from ...sql import func
from ...sql import null
from ...sql import or_
from ...sql import select
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.visitors import InternalTraversal
from ...types import BLOB
from ...types import CHAR
from ...types import CLOB
from ...types import DOUBLE_PRECISION
from ...types import INTEGER
from ...types import NCHAR
from ...types import NVARCHAR
from ...types import REAL
from ...types import VARCHAR
def _parse_identity_options(self, identity_options, default_on_null):
    parts = [p.strip() for p in identity_options.split(',')]
    identity = {'always': parts[0] == 'ALWAYS', 'on_null': default_on_null == 'YES'}
    for part in parts[1:]:
        option, value = part.split(':')
        value = value.strip()
        if 'START WITH' in option:
            identity['start'] = int(value)
        elif 'INCREMENT BY' in option:
            identity['increment'] = int(value)
        elif 'MAX_VALUE' in option:
            identity['maxvalue'] = int(value)
        elif 'MIN_VALUE' in option:
            identity['minvalue'] = int(value)
        elif 'CYCLE_FLAG' in option:
            identity['cycle'] = value == 'Y'
        elif 'CACHE_SIZE' in option:
            identity['cache'] = int(value)
        elif 'ORDER_FLAG' in option:
            identity['order'] = value == 'Y'
    return identity