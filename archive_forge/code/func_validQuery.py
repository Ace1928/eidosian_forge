import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
def validQuery(self, portOnServer, portOnClient):
    """
        Called when a valid query is received to look up and deliver the
        response.

        @param portOnServer: The server port from the query.
        @param portOnClient: The client port from the query.
        """
    serverAddr = (self.transport.getHost().host, portOnServer)
    clientAddr = (self.transport.getPeer().host, portOnClient)
    defer.maybeDeferred(self.lookup, serverAddr, clientAddr).addCallback(self._cbLookup, portOnServer, portOnClient).addErrback(self._ebLookup, portOnServer, portOnClient)