from __future__ import annotations
import logging
import re
from typing import cast
from typing import TYPE_CHECKING
from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import INTERVAL
from .base import PGCompiler
from .base import PGIdentifierPreparer
from .base import REGCONFIG
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .types import CITEXT
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...sql import sqltypes
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class _PsycopgMultiRange(ranges.AbstractMultiRangeImpl):

    def bind_processor(self, dialect):
        psycopg_Range = cast(PGDialect_psycopg, dialect)._psycopg_Range
        psycopg_Multirange = cast(PGDialect_psycopg, dialect)._psycopg_Multirange
        NoneType = type(None)

        def to_range(value):
            if isinstance(value, (str, NoneType, psycopg_Multirange)):
                return value
            return psycopg_Multirange([psycopg_Range(element.lower, element.upper, element.bounds, element.empty) for element in cast('Iterable[ranges.Range]', value)])
        return to_range

    def result_processor(self, dialect, coltype):

        def to_range(value):
            if value is None:
                return None
            else:
                return ranges.MultiRange((ranges.Range(elem._lower, elem._upper, bounds=elem._bounds if elem._bounds else '[)', empty=not elem._bounds) for elem in value))
        return to_range