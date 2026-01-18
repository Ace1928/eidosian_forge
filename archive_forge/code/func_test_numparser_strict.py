from __future__ import absolute_import, print_function, division
from petl.compat import maxint
from petl.test.helpers import eq_
from petl.util.parsers import numparser, datetimeparser
def test_numparser_strict():
    parsenumber = numparser(strict=True)
    assert parsenumber('1') == 1
    assert parsenumber('1.0') == 1.0
    assert parsenumber(str(maxint + 1)) == maxint + 1
    assert parsenumber('3+4j') == 3 + 4j
    try:
        parsenumber('aaa')
    except ValueError:
        pass
    else:
        assert False, 'expected exception'
    try:
        parsenumber(None)
    except TypeError:
        pass
    else:
        assert False, 'expected exception'