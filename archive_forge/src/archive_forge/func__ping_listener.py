import copy
import functools
import threading
import time
from oslo_utils import strutils
import sqlalchemy as sa
from sqlalchemy import exc as sa_exc
from sqlalchemy import pool as sa_pool
from sqlalchemy import sql
import tenacity
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence.backends.sqlalchemy import migration
from taskflow.persistence.backends.sqlalchemy import tables
from taskflow.persistence import base
from taskflow.persistence import models
from taskflow.utils import eventlet_utils
from taskflow.utils import misc
def _ping_listener(dbapi_conn, connection_rec, connection_proxy):
    """Ensures that MySQL connections checked out of the pool are alive.

    Modified + borrowed from: http://bit.ly/14BYaW6.
    """
    try:
        dbapi_conn.cursor().execute('select 1')
    except dbapi_conn.OperationalError as ex:
        if _in_any(str(ex.args[0]), MY_SQL_GONE_WAY_AWAY_ERRORS):
            LOG.warning('Got mysql server has gone away', exc_info=True)
            raise sa_exc.DisconnectionError('Database server went away')
        elif _in_any(str(ex.args[0]), POSTGRES_GONE_WAY_AWAY_ERRORS):
            LOG.warning('Got postgres server has gone away', exc_info=True)
            raise sa_exc.DisconnectionError('Database server went away')
        else:
            raise