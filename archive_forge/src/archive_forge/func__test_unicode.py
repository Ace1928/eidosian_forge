from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq
def _test_unicode(dbo):
    expect = ((u'name', u'id'), (u'Արամ Խաչատրյան', 1), (u'Johann Strauß', 2), (u'Вагиф Сәмәдоғлу', 3), (u'章子怡', 4))
    actual = etl.fromdb(dbo, 'SELECT * FROM test_unicode')
    print('write some data and verify...')
    etl.todb(expect, dbo, 'test_unicode')
    ieq(expect, actual)
    print(etl.look(actual))