from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import sqlite3
from petl.test.helpers import ieq
from petl.io.db import fromdb, todb, appenddb
def test_tosqlite3_appendsqlite3_connection():
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE foobar (foo TEXT, bar INT)')
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    todb(table, conn, 'foobar')
    actual = conn.execute('SELECT * FROM foobar')
    expect = (('a', 1), ('b', 2), ('c', 2))
    ieq(expect, actual)
    table2 = (('foo', 'bar'), ('d', 7), ('e', 9), ('f', 1))
    appenddb(table2, conn, 'foobar')
    actual = conn.execute('SELECT * FROM foobar')
    expect = (('a', 1), ('b', 2), ('c', 2), ('d', 7), ('e', 9), ('f', 1))
    ieq(expect, actual)