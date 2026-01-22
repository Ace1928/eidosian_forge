from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
class PBConnTestCase(unittest.TestCase):
    unsafeTracebacks = 0

    def setUp(self):
        self.serverFactory = SaveProtocolServerFactory(SimpleRoot())
        self.serverFactory.unsafeTracebacks = self.unsafeTracebacks
        self.clientFactory = pb.PBClientFactory()
        self.connectedServer, self.connectedClient, self.pump = connectedServerAndClient(lambda: self.serverFactory.buildProtocol(None), lambda: self.clientFactory.buildProtocol(None))