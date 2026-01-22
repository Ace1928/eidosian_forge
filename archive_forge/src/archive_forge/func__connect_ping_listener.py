import functools
import itertools
import logging
import os
import re
import time
import debtcollector.removals
import debtcollector.renames
import sqlalchemy
from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy import pool
from sqlalchemy import select
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
def _connect_ping_listener(connection, branch):
    """Ping the server at connection startup.

    Ping the server at transaction begin and transparently reconnect
    if a disconnect exception occurs.

    This listener is used up until SQLAlchemy 2.0.5.  At 2.0.5, we use the
    ``pool_pre_ping`` parameter instead of this event handler.

    Note the current test suite in test_exc_filters still **tests** this
    handler using all SQLAlchemy versions including 2.0.5 and greater.

    """
    if branch:
        return
    save_should_close_with_result = connection.should_close_with_result
    connection.should_close_with_result = False
    try:
        connection.scalar(select(1))
    except exception.DBConnectionError:
        LOG.exception('Database connection was found disconnected; reconnecting')
        if hasattr(connection, 'rollback'):
            connection.rollback()
        connection.scalar(select(1))
    finally:
        connection.should_close_with_result = save_should_close_with_result
        if hasattr(connection, 'rollback'):
            connection.rollback()