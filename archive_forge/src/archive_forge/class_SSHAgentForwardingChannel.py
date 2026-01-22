import os
from twisted.conch.ssh import agent, channel, keys
from twisted.internet import protocol, reactor
from twisted.logger import Logger
class SSHAgentForwardingChannel(channel.SSHChannel):

    def channelOpen(self, specificData):
        cc = protocol.ClientCreator(reactor, SSHAgentForwardingLocal)
        d = cc.connectUNIX(os.environ['SSH_AUTH_SOCK'])
        d.addCallback(self._cbGotLocal)
        d.addErrback(lambda x: self.loseConnection())
        self.buf = ''

    def _cbGotLocal(self, local):
        self.local = local
        self.dataReceived = self.local.transport.write
        self.local.dataReceived = self.write

    def dataReceived(self, data):
        self.buf += data

    def closed(self):
        if self.local:
            self.local.loseConnection()
            self.local = None