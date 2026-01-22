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
def _do_prepared_twophase(self, connection, command, recover=False):
    dbapi_conn = connection.connection.dbapi_connection
    if recover or dbapi_conn.info.transaction_status != self._psycopg_TransactionStatus.IDLE:
        dbapi_conn.rollback()
    before_autocommit = dbapi_conn.autocommit
    try:
        if not before_autocommit:
            self._do_autocommit(dbapi_conn, True)
        dbapi_conn.execute(command)
    finally:
        if not before_autocommit:
            self._do_autocommit(dbapi_conn, before_autocommit)