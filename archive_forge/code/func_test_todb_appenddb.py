from __future__ import absolute_import, print_function, division
import sqlite3
from tempfile import NamedTemporaryFile
from petl.compat import next
from petl.test.helpers import ieq, eq_
from petl.io.db import fromdb, todb, appenddb
def test_todb_appenddb():
    f = NamedTemporaryFile(delete=False)
    conn = sqlite3.connect(f.name)
    conn.execute('create table foobar (foo, bar)')
    conn.commit()
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    todb(table, conn, 'foobar')
    actual = conn.execute('select * from foobar')
    expect = (('a', 1), ('b', 2), ('c', 2))
    ieq(expect, actual)
    table2 = (('foo', 'bar'), ('d', 7), ('e', 9), ('f', 1))
    appenddb(table2, conn, 'foobar')
    actual = conn.execute('select * from foobar')
    expect = (('a', 1), ('b', 2), ('c', 2), ('d', 7), ('e', 9), ('f', 1))
    ieq(expect, actual)