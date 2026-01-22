import decimal
import re
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .pg_catalog import _SpaceVector
from .pg_catalog import OIDVECTOR
from .types import CITEXT
from ... import exc
from ... import util
from ...engine import processors
from ...sql import sqltypes
from ...sql.elements import quoted_name
class _Pg8000MultiRange(ranges.AbstractMultiRangeImpl):

    def bind_processor(self, dialect):
        pg8000_Range = dialect.dbapi.Range

        def to_multirange(value):
            if isinstance(value, list):
                mr = []
                for v in value:
                    if isinstance(v, ranges.Range):
                        mr.append(pg8000_Range(v.lower, v.upper, v.bounds, v.empty))
                    else:
                        mr.append(v)
                return mr
            else:
                return value
        return to_multirange

    def result_processor(self, dialect, coltype):

        def to_multirange(value):
            if value is None:
                return None
            else:
                return ranges.MultiRange((ranges.Range(v.lower, v.upper, bounds=v.bounds, empty=v.is_empty) for v in value))
        return to_multirange