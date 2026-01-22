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
class PGInspector(reflection.Inspector):
    dialect: PGDialect

    def get_table_oid(self, table_name: str, schema: Optional[str]=None) -> int:
        """Return the OID for the given table name.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        """
        with self._operation_context() as conn:
            return self.dialect.get_table_oid(conn, table_name, schema, info_cache=self.info_cache)

    def get_domains(self, schema: Optional[str]=None) -> List[ReflectedDomain]:
        """Return a list of DOMAIN objects.

        Each member is a dictionary containing these fields:

            * name - name of the domain
            * schema - the schema name for the domain.
            * visible - boolean, whether or not this domain is visible
              in the default search path.
            * type - the type defined by this domain.
            * nullable - Indicates if this domain can be ``NULL``.
            * default - The default value of the domain or ``None`` if the
              domain has no default.
            * constraints - A list of dict wit the constraint defined by this
              domain. Each element constaints two keys: ``name`` of the
              constraint and ``check`` with the constraint text.

        :param schema: schema name.  If None, the default schema
         (typically 'public') is used.  May also be set to ``'*'`` to
         indicate load domains for all schemas.

        .. versionadded:: 2.0

        """
        with self._operation_context() as conn:
            return self.dialect._load_domains(conn, schema, info_cache=self.info_cache)

    def get_enums(self, schema: Optional[str]=None) -> List[ReflectedEnum]:
        """Return a list of ENUM objects.

        Each member is a dictionary containing these fields:

            * name - name of the enum
            * schema - the schema name for the enum.
            * visible - boolean, whether or not this enum is visible
              in the default search path.
            * labels - a list of string labels that apply to the enum.

        :param schema: schema name.  If None, the default schema
         (typically 'public') is used.  May also be set to ``'*'`` to
         indicate load enums for all schemas.

        """
        with self._operation_context() as conn:
            return self.dialect._load_enums(conn, schema, info_cache=self.info_cache)

    def get_foreign_table_names(self, schema: Optional[str]=None) -> List[str]:
        """Return a list of FOREIGN TABLE names.

        Behavior is similar to that of
        :meth:`_reflection.Inspector.get_table_names`,
        except that the list is limited to those tables that report a
        ``relkind`` value of ``f``.

        """
        with self._operation_context() as conn:
            return self.dialect._get_foreign_table_names(conn, schema, info_cache=self.info_cache)

    def has_type(self, type_name: str, schema: Optional[str]=None, **kw: Any) -> bool:
        """Return if the database has the specified type in the provided
        schema.

        :param type_name: the type to check.
        :param schema: schema name.  If None, the default schema
         (typically 'public') is used.  May also be set to ``'*'`` to
         check in all schemas.

        .. versionadded:: 2.0

        """
        with self._operation_context() as conn:
            return self.dialect.has_type(conn, type_name, schema, info_cache=self.info_cache)