import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
class SSHForwardingClient(protocol.Protocol):

    def __init__(self, channel):
        self.channel = channel
        self.buf = b'\x00'

    def dataReceived(self, data):
        if self.buf:
            self.buf += data
        else:
            self.channel.write(data)

    def connectionLost(self, reason):
        if self.channel:
            self.channel.loseConnection()
            self.channel = None