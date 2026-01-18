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

.. dialect:: postgresql+psycopg
    :name: psycopg (a.k.a. psycopg 3)
    :dbapi: psycopg
    :connectstring: postgresql+psycopg://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://pypi.org/project/psycopg/

``psycopg`` is the package and module name for version 3 of the ``psycopg``
database driver, formerly known as ``psycopg2``.  This driver is different
enough from its ``psycopg2`` predecessor that SQLAlchemy supports it
via a totally separate dialect; support for ``psycopg2`` is expected to remain
for as long as that package continues to function for modern Python versions,
and also remains the default dialect for the ``postgresql://`` dialect
series.

The SQLAlchemy ``psycopg`` dialect provides both a sync and an async
implementation under the same dialect name. The proper version is
selected depending on how the engine is created:

* calling :func:`_sa.create_engine` with ``postgresql+psycopg://...`` will
  automatically select the sync version, e.g.::

    from sqlalchemy import create_engine
    sync_engine = create_engine("postgresql+psycopg://scott:tiger@localhost/test")

* calling :func:`_asyncio.create_async_engine` with
  ``postgresql+psycopg://...`` will automatically select the async version,
  e.g.::

    from sqlalchemy.ext.asyncio import create_async_engine
    asyncio_engine = create_async_engine("postgresql+psycopg://scott:tiger@localhost/test")

The asyncio version of the dialect may also be specified explicitly using the
``psycopg_async`` suffix, as::

    from sqlalchemy.ext.asyncio import create_async_engine
    asyncio_engine = create_async_engine("postgresql+psycopg_async://scott:tiger@localhost/test")

.. seealso::

    :ref:`postgresql_psycopg2` - The SQLAlchemy ``psycopg``
    dialect shares most of its behavior with the ``psycopg2`` dialect.
    Further documentation is available there.

