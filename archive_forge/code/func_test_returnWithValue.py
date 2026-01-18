from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_returnWithValue(self):
    """
        If the C{return} statement has a value it is propagated back to the
        L{Deferred} that the C{inlineCallbacks} function returned.
        """
    environ = {'inlineCallbacks': inlineCallbacks}
    exec('\n@inlineCallbacks\ndef f(d):\n    yield d\n    return 14\n        ', environ)
    d1 = Deferred()
    d2 = environ['f'](d1)
    d1.callback(None)
    self.assertEqual(self.successResultOf(d2), 14)