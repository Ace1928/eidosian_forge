from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_synchronousError(self):
    """
        Like L{test_asynchronousError}, but for a method which synchronously
        raises a L{pb.Error} subclass.
        """
    return self._exceptionTest('synchronousError', SynchronousError, False)