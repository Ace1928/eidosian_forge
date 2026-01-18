from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_synchronousException(self):
    """
        Like L{test_asynchronousException}, but for a method which raises an
        exception synchronously.
        """
    return self._exceptionTest('synchronousException', SynchronousException, True)