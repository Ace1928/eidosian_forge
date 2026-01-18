from __future__ import absolute_import, print_function, division
import logging
from datetime import datetime, date
import sqlite3
import pytest
from petl.io.db import fromdb, todb
from petl.io.db_create import make_sqlalchemy_column
from petl.test.helpers import ieq, eq_
from petl.util.vis import look
from petl.test.io.test_db_server import user, password, host, database
def test_sqlite3_create():
    dbapi_connection = sqlite3.connect(':memory:')
    _setup_generic(dbapi_connection)
    _test_create(dbapi_connection)
    _setup_generic(dbapi_connection)
    dbapi_cursor = dbapi_connection.cursor()
    _test_create(dbapi_cursor)
    dbapi_cursor.close()