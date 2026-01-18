import inspect
import unittest
from traits.api import (
def test_override_validate(self):
    """ Verify `BaseCallable` can be subclassed to create new traits.
        """

    class ZeroArgsCallable(BaseCallable):

        def validate(self, object, name, value):
            if callable(value):
                sig = inspect.signature(value)
                if len(sig.parameters) == 0:
                    return value
            self.error(object, name, value)

    class Foo(HasTraits):
        value = ZeroArgsCallable
    Foo(value=lambda: 1)
    with self.assertRaises(TraitError):
        Foo(value=lambda x: x)
    with self.assertRaises(TraitError):
        Foo(value=1)