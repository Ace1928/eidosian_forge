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
def _split_multihost_from_url(self, url: URL) -> Union[Tuple[None, None], Tuple[Tuple[Optional[str], ...], Tuple[Optional[int], ...]]]:
    hosts: Optional[Tuple[Optional[str], ...]] = None
    ports_str: Union[str, Tuple[Optional[str], ...], None] = None
    integrated_multihost = False
    if 'host' in url.query:
        if isinstance(url.query['host'], (list, tuple)):
            integrated_multihost = True
            hosts, ports_str = zip(*[token.split(':') if ':' in token else (token, None) for token in url.query['host']])
        elif isinstance(url.query['host'], str):
            hosts = tuple(url.query['host'].split(','))
            if 'port' not in url.query and len(hosts) == 1 and (':' in hosts[0]):
                host_port_match = re.match('^([a-zA-Z0-9\\-\\.]*)(?:\\:(\\d*))?$', hosts[0])
                if host_port_match:
                    integrated_multihost = True
                    h, p = host_port_match.group(1, 2)
                    if TYPE_CHECKING:
                        assert isinstance(h, str)
                        assert isinstance(p, str)
                    hosts = (h,)
                    ports_str = cast('Tuple[Optional[str], ...]', (p,) if p else (None,))
    if 'port' in url.query:
        if integrated_multihost:
            raise exc.ArgumentError('Can\'t mix \'multihost\' formats together; use "host=h1,h2,h3&port=p1,p2,p3" or "host=h1:p1&host=h2:p2&host=h3:p3" separately')
        if isinstance(url.query['port'], (list, tuple)):
            ports_str = url.query['port']
        elif isinstance(url.query['port'], str):
            ports_str = tuple(url.query['port'].split(','))
    ports: Optional[Tuple[Optional[int], ...]] = None
    if ports_str:
        try:
            ports = tuple((int(x) if x else None for x in ports_str))
        except ValueError:
            raise exc.ArgumentError(f'Received non-integer port arguments: {ports_str}') from None
    if ports and (not hosts and len(ports) > 1 or (hosts and ports and (len(hosts) != len(ports)) and (len(hosts) > 1 or len(ports) > 1))):
        raise exc.ArgumentError("number of hosts and ports don't match")
    if hosts is not None:
        if ports is None:
            ports = tuple((None for _ in hosts))
    return (hosts, ports)