import inspect
from .. import decorators, lock
from . import TestCase
class TestOnlyRaisesDecorator(TestCase):

    def raise_ZeroDivisionError(self):
        1 / 0

    def test_raises_approved_error(self):
        decorator = decorators.only_raises(ZeroDivisionError)
        decorated_meth = decorator(self.raise_ZeroDivisionError)
        self.assertRaises(ZeroDivisionError, decorated_meth)

    def test_quietly_logs_unapproved_errors(self):
        decorator = decorators.only_raises(IOError)
        decorated_meth = decorator(self.raise_ZeroDivisionError)
        self.assertLogsError(ZeroDivisionError, decorated_meth)