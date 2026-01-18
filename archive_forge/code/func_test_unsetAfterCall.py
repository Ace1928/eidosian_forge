from twisted.python import context
from twisted.trial.unittest import SynchronousTestCase
def test_unsetAfterCall(self):
    """
        After a L{twisted.python.context.call} completes, keys specified in the
        call are no longer associated with the values from that call.
        """
    context.call({'x': 'y'}, lambda: None)
    self.assertIsNone(context.get('x'))