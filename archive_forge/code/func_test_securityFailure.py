from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_securityFailure(self):
    """
        Test that even if an exception is not explicitly jellyable (by being
        a L{pb.Jellyable} subclass), as long as it is an L{pb.Error}
        subclass it receives the same special treatment.
        """

    def failureSecurity(fail):
        fail.trap(SecurityError)
        self.assertNotIsInstance(fail.type, str)
        self.assertIsInstance(fail.value, fail.type)
        return 4300
    return self._testImpl('security', 4300, failureSecurity)