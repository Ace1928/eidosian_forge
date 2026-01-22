from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
class DummyInvoker:
    """
    A behaviorless object to be used as the invoker parameter to
    L{jelly.jelly}.
    """
    serializingPerspective = None