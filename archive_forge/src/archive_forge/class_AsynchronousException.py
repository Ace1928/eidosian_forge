from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
class AsynchronousException(Exception):
    """
    Helper used to test remote methods which return Deferreds which fail with
    exceptions which are not L{pb.Error} subclasses.
    """