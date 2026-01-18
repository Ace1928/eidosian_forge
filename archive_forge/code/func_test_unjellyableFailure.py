from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_unjellyableFailure(self):
    """
        A non-jellyable L{pb.Error} subclass raised by a remote method is
        turned into a Failure with a type set to the FQPN of the exception
        type.
        """

    def failureUnjellyable(fail):
        self.assertEqual(fail.type, b'twisted.spread.test.test_pbfailure.SynchronousError')
        return 431
    return self._testImpl('synchronousError', 431, failureUnjellyable)