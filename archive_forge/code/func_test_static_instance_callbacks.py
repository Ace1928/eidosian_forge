import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_static_instance_callbacks(self):

    class A(object):

        @staticmethod
        def b(a, b, c):
            pass
    a = A()
    b = A()
    self.assertTrue(reflection.is_same_callback(a.b, b.b))