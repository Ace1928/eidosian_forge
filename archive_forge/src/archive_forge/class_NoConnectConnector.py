import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
class NoConnectConnector:

    def stopConnecting(self):
        raise RuntimeError("Shouldn't be called, we're connected.")

    def connect(self):
        raise RuntimeError("Shouldn't be reconnecting.")