import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
def test_property_there(self):

    class Foo(object):

        @property
        def attribute(self):
            return None
    foo = Foo()
    self.assertEqual(True, safe_hasattr(foo, 'attribute'))