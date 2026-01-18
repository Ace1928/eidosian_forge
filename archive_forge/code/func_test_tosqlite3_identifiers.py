from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import sqlite3
from petl.test.helpers import ieq
from petl.io.db import fromdb, todb, appenddb
def test_tosqlite3_identifiers():
    table = (('foo foo', 'bar.baz.spong`'), ('a', 1), ('b', 2), ('c', 2))
    f = NamedTemporaryFile(delete=False)
    f.close()
    conn = sqlite3.connect(f.name)
    conn.execute('CREATE TABLE "foo "" bar`" ("foo foo" TEXT, "bar.baz.spong`" INT)')
    conn.close()
    todb(table, f.name, 'foo " bar`')
    conn = sqlite3.connect(f.name)
    actual = conn.execute('SELECT * FROM `foo " bar```')
    expect = (('a', 1), ('b', 2), ('c', 2))
    ieq(expect, actual)