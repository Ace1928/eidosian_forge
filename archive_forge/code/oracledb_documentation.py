from __future__ import annotations
import collections
import re
from typing import Any
from typing import TYPE_CHECKING
from .cx_oracle import OracleDialect_cx_oracle as _OracleDialect_cx_oracle
from ... import exc
from ... import pool
from ...connectors.asyncio import AsyncAdapt_dbapi_connection
from ...connectors.asyncio import AsyncAdapt_dbapi_cursor
from ...connectors.asyncio import AsyncAdaptFallback_dbapi_connection
from ...util import asbool
from ...util import await_fallback
from ...util import await_only

.. dialect:: oracle+oracledb
    :name: python-oracledb
    :dbapi: oracledb
    :connectstring: oracle+oracledb://user:pass@hostname:port[/dbname][?service_name=<service>[&key=value&key=value...]]
    :url: https://oracle.github.io/python-oracledb/

python-oracledb is released by Oracle to supersede the cx_Oracle driver.
It is fully compatible with cx_Oracle and features both a "thin" client
mode that requires no dependencies, as well as a "thick" mode that uses
the Oracle Client Interface in the same way as cx_Oracle.

.. seealso::

    :ref:`cx_oracle` - all of cx_Oracle's notes apply to the oracledb driver
    as well.

The SQLAlchemy ``oracledb`` dialect provides both a sync and an async
implementation under the same dialect name. The proper version is
selected depending on how the engine is created:

* calling :func:`_sa.create_engine` with ``oracle+oracledb://...`` will
  automatically select the sync version, e.g.::

    from sqlalchemy import create_engine
    sync_engine = create_engine("oracle+oracledb://scott:tiger@localhost/?service_name=XEPDB1")

* calling :func:`_asyncio.create_async_engine` with
  ``oracle+oracledb://...`` will automatically select the async version,
  e.g.::

    from sqlalchemy.ext.asyncio import create_async_engine
    asyncio_engine = create_async_engine("oracle+oracledb://scott:tiger@localhost/?service_name=XEPDB1")

The asyncio version of the dialect may also be specified explicitly using the
``oracledb_async`` suffix, as::

    from sqlalchemy.ext.asyncio import create_async_engine
    asyncio_engine = create_async_engine("oracle+oracledb_async://scott:tiger@localhost/?service_name=XEPDB1")

.. versionadded:: 2.0.25 added support for the async version of oracledb.

Thick mode support
------------------

By default the ``python-oracledb`` is started in thin mode, that does not
require oracle client libraries to be installed in the system. The
``python-oracledb`` driver also support a "thick" mode, that behaves
similarly to ``cx_oracle`` and requires that Oracle Client Interface (OCI)
is installed.

To enable this mode, the user may call ``oracledb.init_oracle_client``
manually, or by passing the parameter ``thick_mode=True`` to
:func:`_sa.create_engine`. To pass custom arguments to ``init_oracle_client``,
like the ``lib_dir`` path, a dict may be passed to this parameter, as in::

    engine = sa.create_engine("oracle+oracledb://...", thick_mode={
        "lib_dir": "/path/to/oracle/client/lib", "driver_name": "my-app"
    })

.. seealso::

    https://python-oracledb.readthedocs.io/en/latest/api_manual/module.html#oracledb.init_oracle_client


.. versionadded:: 2.0.0 added support for oracledb driver.

