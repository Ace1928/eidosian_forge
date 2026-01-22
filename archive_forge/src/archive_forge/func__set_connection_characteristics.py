from __future__ import annotations
import functools
import operator
import random
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import characteristics
from . import cursor as _cursor
from . import interfaces
from .base import Connection
from .interfaces import CacheStats
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .reflection import ObjectKind
from .reflection import ObjectScope
from .. import event
from .. import exc
from .. import pool
from .. import util
from ..sql import compiler
from ..sql import dml
from ..sql import expression
from ..sql import type_api
from ..sql._typing import is_tuple_type
from ..sql.base import _NoArg
from ..sql.compiler import DDLCompiler
from ..sql.compiler import InsertmanyvaluesSentinelOpts
from ..sql.compiler import SQLCompiler
from ..sql.elements import quoted_name
from ..util.typing import Final
from ..util.typing import Literal
def _set_connection_characteristics(self, connection, characteristics):
    characteristic_values = [(name, self.connection_characteristics[name], value) for name, value in characteristics.items()]
    if connection.in_transaction():
        trans_objs = [(name, obj) for name, obj, value in characteristic_values if obj.transactional]
        if trans_objs:
            raise exc.InvalidRequestError('This connection has already initialized a SQLAlchemy Transaction() object via begin() or autobegin; %s may not be altered unless rollback() or commit() is called first.' % ', '.join((name for name, obj in trans_objs)))
    dbapi_connection = connection.connection.dbapi_connection
    for name, characteristic, value in characteristic_values:
        characteristic.set_characteristic(self, dbapi_connection, value)
    connection.connection._connection_record.finalize_callback.append(functools.partial(self._reset_characteristics, characteristics))