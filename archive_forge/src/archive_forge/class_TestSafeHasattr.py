import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
class TestSafeHasattr(TestCase):

    def test_attribute_not_there(self):

        class Foo(object):
            pass
        self.assertEqual(False, safe_hasattr(Foo(), 'anything'))

    def test_attribute_there(self):

        class Foo(object):
            pass
        foo = Foo()
        foo.attribute = None
        self.assertEqual(True, safe_hasattr(foo, 'attribute'))

    def test_property_there(self):

        class Foo(object):

            @property
            def attribute(self):
                return None
        foo = Foo()
        self.assertEqual(True, safe_hasattr(foo, 'attribute'))

    def test_property_raises(self):

        class Foo(object):

            @property
            def attribute(self):
                1 / 0
        foo = Foo()
        self.assertRaises(ZeroDivisionError, safe_hasattr, foo, 'attribute')