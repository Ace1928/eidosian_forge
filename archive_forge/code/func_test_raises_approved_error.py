import inspect
from .. import decorators, lock
from . import TestCase
def test_raises_approved_error(self):
    decorator = decorators.only_raises(ZeroDivisionError)
    decorated_meth = decorator(self.raise_ZeroDivisionError)
    self.assertRaises(ZeroDivisionError, decorated_meth)