from the proposed insertion.   These values are normally specified using
from __future__ import annotations
from array import array as _array
from collections import defaultdict
from itertools import compress
import re
from typing import cast
from . import reflection as _reflection
from .enumerated import ENUM
from .enumerated import SET
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from .reserved_words import RESERVED_WORDS_MARIADB
from .reserved_words import RESERVED_WORDS_MYSQL
from .types import _FloatType
from .types import _IntegerType
from .types import _MatchType
from .types import _NumericType
from .types import _StringType
from .types import BIGINT
from .types import BIT
from .types import CHAR
from .types import DATETIME
from .types import DECIMAL
from .types import DOUBLE
from .types import FLOAT
from .types import INTEGER
from .types import LONGBLOB
from .types import LONGTEXT
from .types import MEDIUMBLOB
from .types import MEDIUMINT
from .types import MEDIUMTEXT
from .types import NCHAR
from .types import NUMERIC
from .types import NVARCHAR
from .types import REAL
from .types import SMALLINT
from .types import TEXT
from .types import TIME
from .types import TIMESTAMP
from .types import TINYBLOB
from .types import TINYINT
from .types import TINYTEXT
from .types import VARCHAR
from .types import YEAR
from ... import exc
from ... import literal_column
from ... import log
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import functions
from ...sql import operators
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.compiler import SQLCompiler
from ...sql.schema import SchemaConst
from ...types import BINARY
from ...types import BLOB
from ...types import BOOLEAN
from ...types import DATE
from ...types import UUID
from ...types import VARBINARY
from ...util import topological
def visit_typeclause(self, typeclause, type_=None, **kw):
    if type_ is None:
        type_ = typeclause.type.dialect_impl(self.dialect)
    if isinstance(type_, sqltypes.TypeDecorator):
        return self.visit_typeclause(typeclause, type_.impl, **kw)
    elif isinstance(type_, sqltypes.Integer):
        if getattr(type_, 'unsigned', False):
            return 'UNSIGNED INTEGER'
        else:
            return 'SIGNED INTEGER'
    elif isinstance(type_, sqltypes.TIMESTAMP):
        return 'DATETIME'
    elif isinstance(type_, (sqltypes.DECIMAL, sqltypes.DateTime, sqltypes.Date, sqltypes.Time)):
        return self.dialect.type_compiler_instance.process(type_)
    elif isinstance(type_, sqltypes.String) and (not isinstance(type_, (ENUM, SET))):
        adapted = CHAR._adapt_string_for_cast(type_)
        return self.dialect.type_compiler_instance.process(adapted)
    elif isinstance(type_, sqltypes._Binary):
        return 'BINARY'
    elif isinstance(type_, sqltypes.JSON):
        return 'JSON'
    elif isinstance(type_, sqltypes.NUMERIC):
        return self.dialect.type_compiler_instance.process(type_).replace('NUMERIC', 'DECIMAL')
    elif isinstance(type_, sqltypes.Float) and self.dialect._support_float_cast:
        return self.dialect.type_compiler_instance.process(type_)
    else:
        return None