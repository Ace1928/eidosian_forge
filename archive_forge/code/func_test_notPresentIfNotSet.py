from twisted.python import context
from twisted.trial.unittest import SynchronousTestCase
def test_notPresentIfNotSet(self):
    """
        Arbitrary keys which have not been set in the context have an associated
        value of L{None}.
        """
    self.assertIsNone(context.get('x'))