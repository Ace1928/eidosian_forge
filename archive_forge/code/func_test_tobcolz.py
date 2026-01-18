from __future__ import absolute_import, print_function, division
import tempfile
import pytest
from petl.test.helpers import ieq, eq_
from petl.io.bcolz import frombcolz, tobcolz, appendbcolz
def test_tobcolz():
    t = [('foo', 'bar', 'baz'), ('apples', 1, 2.5), ('oranges', 3, 4.4), ('pears', 7, 0.1)]
    ctbl = tobcolz(t)
    assert isinstance(ctbl, bcolz.ctable)
    eq_(t[0], tuple(ctbl.names))
    ieq(t[1:], (tuple(r) for r in ctbl.iter()))
    ctbl = tobcolz(t, chunklen=2)
    assert isinstance(ctbl, bcolz.ctable)
    eq_(t[0], tuple(ctbl.names))
    ieq(t[1:], (tuple(r) for r in ctbl.iter()))
    eq_(2, ctbl.cols[ctbl.names[0]].chunklen)