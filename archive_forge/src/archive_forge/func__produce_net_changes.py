from __future__ import annotations
import contextlib
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import event
from sqlalchemy import inspect
from sqlalchemy import schema as sa_schema
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.sql import expression
from sqlalchemy.sql.schema import ForeignKeyConstraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.sql.schema import UniqueConstraint
from sqlalchemy.util import OrderedSet
from .. import util
from ..ddl._autogen import is_index_sig
from ..ddl._autogen import is_uq_sig
from ..operations import ops
from ..util import sqla_compat
def _produce_net_changes(autogen_context: AutogenContext, upgrade_ops: UpgradeOps) -> None:
    connection = autogen_context.connection
    assert connection is not None
    include_schemas = autogen_context.opts.get('include_schemas', False)
    inspector: Inspector = inspect(connection)
    default_schema = connection.dialect.default_schema_name
    schemas: Set[Optional[str]]
    if include_schemas:
        schemas = set(inspector.get_schema_names())
        schemas.discard('information_schema')
        schemas.discard(default_schema)
        schemas.add(None)
    else:
        schemas = {None}
    schemas = {s for s in schemas if autogen_context.run_name_filters(s, 'schema', {})}
    assert autogen_context.dialect is not None
    comparators.dispatch('schema', autogen_context.dialect.name)(autogen_context, upgrade_ops, schemas)