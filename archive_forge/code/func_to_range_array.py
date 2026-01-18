from __future__ import annotations
import collections
import decimal
import json as _py_json
import re
import time
from . import json
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import OID
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .base import REGCLASS
from .base import REGCONFIG
from .types import BIT
from .types import BYTEA
from .types import CITEXT
from ... import exc
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...engine import processors
from ...sql import sqltypes
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
def to_range_array(value):

    def to_range(rvalue):
        if rvalue is not None:
            empty = rvalue.isempty
            rvalue = ranges.Range(rvalue.lower, rvalue.upper, bounds=f'{('[' if empty or rvalue.lower_inc else '(')}{(']' if not empty and rvalue.upper_inc else ')')}', empty=empty)
        return rvalue
    if value is not None:
        value = ranges.MultiRange((to_range(elem) for elem in value))
    return value