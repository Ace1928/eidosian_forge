import inspect
from .. import decorators, lock
from . import TestCase
def test_quietly_logs_unapproved_errors(self):
    decorator = decorators.only_raises(IOError)
    decorated_meth = decorator(self.raise_ZeroDivisionError)
    self.assertLogsError(ZeroDivisionError, decorated_meth)