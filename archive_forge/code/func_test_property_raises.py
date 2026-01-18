import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_property_raises(self):

    class Foo(object):

        @property
        def attribute(self):
            1 / 0
    foo = Foo()
    self.assertRaises(ZeroDivisionError, safe_hasattr, foo, 'attribute')