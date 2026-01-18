from twisted.python import context
from twisted.trial.unittest import SynchronousTestCase
def test_setByCall(self):
    """
        Values may be associated with keys by passing them in a dictionary as
        the first argument to L{twisted.python.context.call}.
        """
    self.assertEqual(context.call({'x': 'y'}, context.get, 'x'), 'y')