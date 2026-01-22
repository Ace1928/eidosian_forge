import functools
from oslotest import base as test_base
from oslo_utils import reflection
class CallbackEqualityTest(test_base.BaseTestCase):

    def test_different_simple_callbacks(self):

        def a():
            pass

        def b():
            pass
        self.assertFalse(reflection.is_same_callback(a, b))

    def test_static_instance_callbacks(self):

        class A(object):

            @staticmethod
            def b(a, b, c):
                pass
        a = A()
        b = A()
        self.assertTrue(reflection.is_same_callback(a.b, b.b))

    def test_different_instance_callbacks(self):

        class A(object):

            def b(self):
                pass

            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return not self.__eq__(other)
        b = A()
        c = A()
        self.assertFalse(reflection.is_same_callback(b.b, c.b))
        self.assertTrue(reflection.is_same_callback(b.b, b.b))