from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq
def _test_dbo(write_dbo, read_dbo=None):
    if read_dbo is None:
        read_dbo = write_dbo
    expect_empty = (('foo', 'bar'),)
    expect = (('foo', 'bar'), ('a', 1), ('b', 2))
    expect_appended = (('foo', 'bar'), ('a', 1), ('b', 2), ('a', 1), ('b', 2))
    actual = etl.fromdb(read_dbo, 'SELECT * FROM test')
    debug('verify empty to start with...')
    debug(etl.look(actual))
    ieq(expect_empty, actual)
    debug('write some data and verify...')
    etl.todb(expect, write_dbo, 'test')
    debug(etl.look(actual))
    ieq(expect, actual)
    debug('append some data and verify...')
    etl.appenddb(expect, write_dbo, 'test')
    debug(etl.look(actual))
    ieq(expect_appended, actual)
    debug('overwrite and verify...')
    etl.todb(expect, write_dbo, 'test')
    debug(etl.look(actual))
    ieq(expect, actual)
    debug('cut, overwrite and verify')
    etl.todb(etl.cut(expect, 'bar', 'foo'), write_dbo, 'test')
    debug(etl.look(actual))
    ieq(expect, actual)
    debug('cut, append and verify')
    etl.appenddb(etl.cut(expect, 'bar', 'foo'), write_dbo, 'test')
    debug(etl.look(actual))
    ieq(expect_appended, actual)
    debug('try a single row')
    etl.todb(etl.head(expect, 1), write_dbo, 'test')
    debug(etl.look(actual))
    ieq(etl.head(expect, 1), actual)