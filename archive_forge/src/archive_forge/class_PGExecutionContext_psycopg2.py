from __future__ import annotations
import collections.abc as collections_abc
import logging
import re
from typing import cast
from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from ... import types as sqltypes
from ... import util
from ...util import FastIntFlag
from ...util import parse_user_argument_for_enum
class PGExecutionContext_psycopg2(_PGExecutionContext_common_psycopg):
    _psycopg2_fetched_rows = None

    def post_exec(self):
        self._log_notices(self.cursor)

    def _log_notices(self, cursor):
        if not cursor.connection.notices or not isinstance(cursor.connection.notices, collections_abc.Iterable):
            return
        for notice in cursor.connection.notices:
            logger.info(notice.rstrip())
        cursor.connection.notices[:] = []