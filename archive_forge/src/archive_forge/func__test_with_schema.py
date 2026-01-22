from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq
def _test_with_schema(dbo, schema):
    expect = (('foo', 'bar'), ('a', 1), ('b', 2))
    expect_appended = (('foo', 'bar'), ('a', 1), ('b', 2), ('a', 1), ('b', 2))
    actual = etl.fromdb(dbo, 'SELECT * FROM test')
    print('write some data and verify...')
    etl.todb(expect, dbo, 'test', schema=schema)
    ieq(expect, actual)
    print(etl.look(actual))
    print('append some data and verify...')
    etl.appenddb(expect, dbo, 'test', schema=schema)
    ieq(expect_appended, actual)
    print(etl.look(actual))