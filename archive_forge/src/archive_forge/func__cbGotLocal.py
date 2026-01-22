import os
from twisted.conch.ssh import agent, channel, keys
from twisted.internet import protocol, reactor
from twisted.logger import Logger
def _cbGotLocal(self, local):
    self.local = local
    self.dataReceived = self.local.transport.write
    self.local.dataReceived = self.write