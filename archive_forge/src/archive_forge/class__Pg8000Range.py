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
class _Pg8000Range(ranges.AbstractSingleRangeImpl):

    def bind_processor(self, dialect):
        pg8000_Range = dialect.dbapi.Range

        def to_range(value):
            if isinstance(value, ranges.Range):
                value = pg8000_Range(value.lower, value.upper, value.bounds, value.empty)
            return value
        return to_range

    def result_processor(self, dialect, coltype):

        def to_range(value):
            if value is not None:
                value = ranges.Range(value.lower, value.upper, bounds=value.bounds, empty=value.is_empty)
            return value
        return to_range