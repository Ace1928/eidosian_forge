from struct import calcsize, pack, unpack
from twisted.protocols.stateful import StatefulProtocol
from twisted.protocols.test import test_basic
from twisted.trial.unittest import TestCase
def _getString(self, msg):
    self.stringReceived(msg)
    return (self._getHeader, 4)