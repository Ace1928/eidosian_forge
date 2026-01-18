from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_subtract_1():
    left = (('begin', 'end', 'label'), (1, 6, 'apple'), (3, 6, 'orange'), (5, 9, 'banana'))
    right = (('start', 'stop', 'foo'), (3, 4, True))
    expect = (('begin', 'end', 'label'), (1, 3, 'apple'), (4, 6, 'apple'), (4, 6, 'orange'), (5, 9, 'banana'))
    actual = intervalsubtract(left, right, lstart='begin', lstop='end', rstart='start', rstop='stop')
    ieq(expect, actual)
    ieq(expect, actual)