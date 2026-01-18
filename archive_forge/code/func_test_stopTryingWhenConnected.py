import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_stopTryingWhenConnected(self):
    """
        If a L{ReconnectingClientFactory} has C{stopTrying} called while it is
        connected, it does not subsequently attempt to reconnect if the
        connection is later lost.
        """

    class NoConnectConnector:

        def stopConnecting(self):
            raise RuntimeError("Shouldn't be called, we're connected.")

        def connect(self):
            raise RuntimeError("Shouldn't be reconnecting.")
    c = ReconnectingClientFactory()
    c.protocol = Protocol
    c.buildProtocol(None)
    c.stopTrying()
    c.clientConnectionLost(NoConnectConnector(), None)
    self.assertFalse(c.continueTrying)