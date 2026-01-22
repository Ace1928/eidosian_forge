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
@util.memoized_property
def _index_query(self):
    pg_class_index = pg_catalog.pg_class.alias('cls_idx')
    idx_sq = select(pg_catalog.pg_index.c.indexrelid, pg_catalog.pg_index.c.indrelid, sql.func.unnest(pg_catalog.pg_index.c.indkey).label('attnum'), sql.func.generate_subscripts(pg_catalog.pg_index.c.indkey, 1).label('ord')).where(~pg_catalog.pg_index.c.indisprimary, pg_catalog.pg_index.c.indrelid.in_(bindparam('oids'))).subquery('idx')
    attr_sq = select(idx_sq.c.indexrelid, idx_sq.c.indrelid, idx_sq.c.ord, sql.case((idx_sq.c.attnum == 0, pg_catalog.pg_get_indexdef(idx_sq.c.indexrelid, idx_sq.c.ord + 1, True)), else_=pg_catalog.pg_attribute.c.attname.cast(TEXT)).label('element'), (idx_sq.c.attnum == 0).label('is_expr')).select_from(idx_sq).outerjoin(pg_catalog.pg_attribute, sql.and_(pg_catalog.pg_attribute.c.attnum == idx_sq.c.attnum, pg_catalog.pg_attribute.c.attrelid == idx_sq.c.indrelid)).where(idx_sq.c.indrelid.in_(bindparam('oids'))).subquery('idx_attr')
    cols_sq = select(attr_sq.c.indexrelid, sql.func.min(attr_sq.c.indrelid), sql.func.array_agg(aggregate_order_by(attr_sq.c.element, attr_sq.c.ord)).label('elements'), sql.func.array_agg(aggregate_order_by(attr_sq.c.is_expr, attr_sq.c.ord)).label('elements_is_expr')).group_by(attr_sq.c.indexrelid).subquery('idx_cols')
    if self.server_version_info >= (11, 0):
        indnkeyatts = pg_catalog.pg_index.c.indnkeyatts
    else:
        indnkeyatts = sql.null().label('indnkeyatts')
    if self.server_version_info >= (15,):
        nulls_not_distinct = pg_catalog.pg_index.c.indnullsnotdistinct
    else:
        nulls_not_distinct = sql.false().label('indnullsnotdistinct')
    return select(pg_catalog.pg_index.c.indrelid, pg_class_index.c.relname.label('relname_index'), pg_catalog.pg_index.c.indisunique, pg_catalog.pg_constraint.c.conrelid.is_not(None).label('has_constraint'), pg_catalog.pg_index.c.indoption, pg_class_index.c.reloptions, pg_catalog.pg_am.c.amname, sql.case((pg_catalog.pg_index.c.indpred.is_not(None), pg_catalog.pg_get_expr(pg_catalog.pg_index.c.indpred, pg_catalog.pg_index.c.indrelid)), else_=None).label('filter_definition'), indnkeyatts, nulls_not_distinct, cols_sq.c.elements, cols_sq.c.elements_is_expr).select_from(pg_catalog.pg_index).where(pg_catalog.pg_index.c.indrelid.in_(bindparam('oids')), ~pg_catalog.pg_index.c.indisprimary).join(pg_class_index, pg_catalog.pg_index.c.indexrelid == pg_class_index.c.oid).join(pg_catalog.pg_am, pg_class_index.c.relam == pg_catalog.pg_am.c.oid).outerjoin(cols_sq, pg_catalog.pg_index.c.indexrelid == cols_sq.c.indexrelid).outerjoin(pg_catalog.pg_constraint, sql.and_(pg_catalog.pg_index.c.indrelid == pg_catalog.pg_constraint.c.conrelid, pg_catalog.pg_index.c.indexrelid == pg_catalog.pg_constraint.c.conindid, pg_catalog.pg_constraint.c.contype == sql.any_(_array.array(('p', 'u', 'x'))))).order_by(pg_catalog.pg_index.c.indrelid, pg_class_index.c.relname)