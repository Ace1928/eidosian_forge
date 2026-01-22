from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq
def _setup_postgresql(dbapi_connection):
    cursor = dbapi_connection.cursor()
    cursor.execute('DROP TABLE IF EXISTS test')
    cursor.execute('CREATE TABLE test (foo TEXT, bar INT)')
    cursor.execute('DROP TABLE IF EXISTS test_unicode')
    cursor.execute('CREATE TABLE test_unicode (name TEXT, id INT)')
    cursor.close()
    dbapi_connection.commit()