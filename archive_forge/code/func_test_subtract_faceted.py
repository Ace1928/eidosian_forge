from __future__ import absolute_import, print_function, division
import logging
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_
from petl.util.vis import lookall
from petl.errors import DuplicateKeyError
from petl.transform.intervals import intervallookup, intervallookupone, \
def test_subtract_faceted():
    left = (('region', 'begin', 'end', 'label'), ('north', 1, 6, 'apple'), ('south', 3, 6, 'orange'), ('west', 5, 9, 'banana'))
    right = (('place', 'start', 'stop', 'foo'), ('south', 3, 4, True), ('north', 5, 6, True))
    expect = (('region', 'begin', 'end', 'label'), ('north', 1, 5, 'apple'), ('south', 4, 6, 'orange'), ('west', 5, 9, 'banana'))
    actual = intervalsubtract(left, right, lkey='region', rkey='place', lstart='begin', lstop='end', rstart='start', rstop='stop')
    ieq(expect, actual)
    ieq(expect, actual)