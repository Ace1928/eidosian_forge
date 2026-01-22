import os
from twisted.internet import fdesc, process, reactor
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.protocols import wire
class InetdFactory(ServerFactory):
    protocol = InetdProtocol
    stderrFile = None

    def __init__(self, service):
        self.service = service