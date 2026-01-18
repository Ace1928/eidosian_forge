from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
def test_register_instance_method():

    class Test:
        __init__ = MethodDispatcher('f')

        @__init__.register(list)
        def _init_list(self, data):
            self.data = data

        @__init__.register(object)
        def _init_obj(self, datum):
            self.data = [datum]
    a = Test(3)
    b = Test([3])
    assert a.data == b.data