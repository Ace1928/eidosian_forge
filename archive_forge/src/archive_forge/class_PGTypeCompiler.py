from the proposed insertion.   These values are specified using the
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import re
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import arraylib as _array
from . import json as _json
from . import pg_catalog
from . import ranges as _ranges
from .ext import _regconfig_fn
from .ext import aggregate_order_by
from .hstore import HSTORE
from .named_types import CreateDomainType as CreateDomainType  # noqa: F401
from .named_types import CreateEnumType as CreateEnumType  # noqa: F401
from .named_types import DOMAIN as DOMAIN  # noqa: F401
from .named_types import DropDomainType as DropDomainType  # noqa: F401
from .named_types import DropEnumType as DropEnumType  # noqa: F401
from .named_types import ENUM as ENUM  # noqa: F401
from .named_types import NamedType as NamedType  # noqa: F401
from .types import _DECIMAL_TYPES  # noqa: F401
from .types import _FLOAT_TYPES  # noqa: F401
from .types import _INT_TYPES  # noqa: F401
from .types import BIT as BIT
from .types import BYTEA as BYTEA
from .types import CIDR as CIDR
from .types import CITEXT as CITEXT
from .types import INET as INET
from .types import INTERVAL as INTERVAL
from .types import MACADDR as MACADDR
from .types import MACADDR8 as MACADDR8
from .types import MONEY as MONEY
from .types import OID as OID
from .types import PGBit as PGBit  # noqa: F401
from .types import PGCidr as PGCidr  # noqa: F401
from .types import PGInet as PGInet  # noqa: F401
from .types import PGInterval as PGInterval  # noqa: F401
from .types import PGMacAddr as PGMacAddr  # noqa: F401
from .types import PGMacAddr8 as PGMacAddr8  # noqa: F401
from .types import PGUuid as PGUuid
from .types import REGCLASS as REGCLASS
from .types import REGCONFIG as REGCONFIG  # noqa: F401
from .types import TIME as TIME
from .types import TIMESTAMP as TIMESTAMP
from .types import TSVECTOR as TSVECTOR
from ... import exc
from ... import schema
from ... import select
from ... import sql
from ... import util
from ...engine import characteristics
from ...engine import default
from ...engine import interfaces
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine import URL
from ...engine.reflection import ReflectionDefaults
from ...sql import bindparam
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.visitors import InternalTraversal
from ...types import BIGINT
from ...types import BOOLEAN
from ...types import CHAR
from ...types import DATE
from ...types import DOUBLE_PRECISION
from ...types import FLOAT
from ...types import INTEGER
from ...types import NUMERIC
from ...types import REAL
from ...types import SMALLINT
from ...types import TEXT
from ...types import UUID as UUID
from ...types import VARCHAR
from ...util.typing import TypedDict
class PGTypeCompiler(compiler.GenericTypeCompiler):

    def visit_TSVECTOR(self, type_, **kw):
        return 'TSVECTOR'

    def visit_TSQUERY(self, type_, **kw):
        return 'TSQUERY'

    def visit_INET(self, type_, **kw):
        return 'INET'

    def visit_CIDR(self, type_, **kw):
        return 'CIDR'

    def visit_CITEXT(self, type_, **kw):
        return 'CITEXT'

    def visit_MACADDR(self, type_, **kw):
        return 'MACADDR'

    def visit_MACADDR8(self, type_, **kw):
        return 'MACADDR8'

    def visit_MONEY(self, type_, **kw):
        return 'MONEY'

    def visit_OID(self, type_, **kw):
        return 'OID'

    def visit_REGCONFIG(self, type_, **kw):
        return 'REGCONFIG'

    def visit_REGCLASS(self, type_, **kw):
        return 'REGCLASS'

    def visit_FLOAT(self, type_, **kw):
        if not type_.precision:
            return 'FLOAT'
        else:
            return 'FLOAT(%(precision)s)' % {'precision': type_.precision}

    def visit_double(self, type_, **kw):
        return self.visit_DOUBLE_PRECISION(type, **kw)

    def visit_BIGINT(self, type_, **kw):
        return 'BIGINT'

    def visit_HSTORE(self, type_, **kw):
        return 'HSTORE'

    def visit_JSON(self, type_, **kw):
        return 'JSON'

    def visit_JSONB(self, type_, **kw):
        return 'JSONB'

    def visit_INT4MULTIRANGE(self, type_, **kw):
        return 'INT4MULTIRANGE'

    def visit_INT8MULTIRANGE(self, type_, **kw):
        return 'INT8MULTIRANGE'

    def visit_NUMMULTIRANGE(self, type_, **kw):
        return 'NUMMULTIRANGE'

    def visit_DATEMULTIRANGE(self, type_, **kw):
        return 'DATEMULTIRANGE'

    def visit_TSMULTIRANGE(self, type_, **kw):
        return 'TSMULTIRANGE'

    def visit_TSTZMULTIRANGE(self, type_, **kw):
        return 'TSTZMULTIRANGE'

    def visit_INT4RANGE(self, type_, **kw):
        return 'INT4RANGE'

    def visit_INT8RANGE(self, type_, **kw):
        return 'INT8RANGE'

    def visit_NUMRANGE(self, type_, **kw):
        return 'NUMRANGE'

    def visit_DATERANGE(self, type_, **kw):
        return 'DATERANGE'

    def visit_TSRANGE(self, type_, **kw):
        return 'TSRANGE'

    def visit_TSTZRANGE(self, type_, **kw):
        return 'TSTZRANGE'

    def visit_json_int_index(self, type_, **kw):
        return 'INT'

    def visit_json_str_index(self, type_, **kw):
        return 'TEXT'

    def visit_datetime(self, type_, **kw):
        return self.visit_TIMESTAMP(type_, **kw)

    def visit_enum(self, type_, **kw):
        if not type_.native_enum or not self.dialect.supports_native_enum:
            return super().visit_enum(type_, **kw)
        else:
            return self.visit_ENUM(type_, **kw)

    def visit_ENUM(self, type_, identifier_preparer=None, **kw):
        if identifier_preparer is None:
            identifier_preparer = self.dialect.identifier_preparer
        return identifier_preparer.format_type(type_)

    def visit_DOMAIN(self, type_, identifier_preparer=None, **kw):
        if identifier_preparer is None:
            identifier_preparer = self.dialect.identifier_preparer
        return identifier_preparer.format_type(type_)

    def visit_TIMESTAMP(self, type_, **kw):
        return 'TIMESTAMP%s %s' % ('(%d)' % type_.precision if getattr(type_, 'precision', None) is not None else '', (type_.timezone and 'WITH' or 'WITHOUT') + ' TIME ZONE')

    def visit_TIME(self, type_, **kw):
        return 'TIME%s %s' % ('(%d)' % type_.precision if getattr(type_, 'precision', None) is not None else '', (type_.timezone and 'WITH' or 'WITHOUT') + ' TIME ZONE')

    def visit_INTERVAL(self, type_, **kw):
        text = 'INTERVAL'
        if type_.fields is not None:
            text += ' ' + type_.fields
        if type_.precision is not None:
            text += ' (%d)' % type_.precision
        return text

    def visit_BIT(self, type_, **kw):
        if type_.varying:
            compiled = 'BIT VARYING'
            if type_.length is not None:
                compiled += '(%d)' % type_.length
        else:
            compiled = 'BIT(%d)' % type_.length
        return compiled

    def visit_uuid(self, type_, **kw):
        if type_.native_uuid:
            return self.visit_UUID(type_, **kw)
        else:
            return super().visit_uuid(type_, **kw)

    def visit_UUID(self, type_, **kw):
        return 'UUID'

    def visit_large_binary(self, type_, **kw):
        return self.visit_BYTEA(type_, **kw)

    def visit_BYTEA(self, type_, **kw):
        return 'BYTEA'

    def visit_ARRAY(self, type_, **kw):
        inner = self.process(type_.item_type, **kw)
        return re.sub('((?: COLLATE.*)?)$', '%s\\1' % ('[]' * (type_.dimensions if type_.dimensions is not None else 1)), inner, count=1)

    def visit_json_path(self, type_, **kw):
        return self.visit_JSONPATH(type_, **kw)

    def visit_JSONPATH(self, type_, **kw):
        return 'JSONPATH'