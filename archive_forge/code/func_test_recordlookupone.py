from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import DuplicateKeyError, FieldSelectionError
from petl.test.helpers import eq_
from petl import cut, lookup, lookupone, dictlookup, dictlookupone, \
def test_recordlookupone():
    t1 = (('foo', 'bar'), ('a', 1), ('b', 2), ('b', 3))
    try:
        recordlookupone(t1, 'foo', strict=True)
    except DuplicateKeyError:
        pass
    else:
        assert False, 'expected error'
    lkp = recordlookupone(t1, 'foo', strict=False)
    eq_('a', lkp['a'].foo)
    eq_('b', lkp['b'].foo)
    eq_(1, lkp['a'].bar)
    eq_(2, lkp['b'].bar)
    lkp = recordlookupone(cut(t1, 'foo'), 'foo', strict=False)
    eq_('a', lkp['a'].foo)
    eq_('b', lkp['b'].foo)