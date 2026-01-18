from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_unknownFailure(self):
    """
        Test that an exception which is a subclass of L{pb.Error} but not
        known on the client side has its type set properly.
        """

    def failureUnknown(fail):
        self.assertEqual(fail.type, b'twisted.spread.test.test_pbfailure.UnknownError')
        return 4310
    return self._testImpl('unknownError', 4310, failureUnknown)