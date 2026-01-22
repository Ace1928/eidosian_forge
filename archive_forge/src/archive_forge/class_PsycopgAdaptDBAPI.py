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
class PsycopgAdaptDBAPI:

    def __init__(self, psycopg) -> None:
        self.psycopg = psycopg
        for k, v in self.psycopg.__dict__.items():
            if k != 'connect':
                self.__dict__[k] = v

    def connect(self, *arg, **kw):
        async_fallback = kw.pop('async_fallback', False)
        creator_fn = kw.pop('async_creator_fn', self.psycopg.AsyncConnection.connect)
        if util.asbool(async_fallback):
            return AsyncAdaptFallback_psycopg_connection(await_fallback(creator_fn(*arg, **kw)))
        else:
            return AsyncAdapt_psycopg_connection(await_only(creator_fn(*arg, **kw)))