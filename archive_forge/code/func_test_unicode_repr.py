from collections import Counter, defaultdict, deque, OrderedDict, UserList
import os
import pytest
import types
import string
import sys
import unittest
import pytest
from IPython.lib import pretty
from io import StringIO
def test_unicode_repr():
    u = u'üniçodé'
    ustr = u

    class C(object):

        def __repr__(self):
            return ustr
    c = C()
    p = pretty.pretty(c)
    assert p == u
    p = pretty.pretty([c])
    assert p == '[%s]' % u