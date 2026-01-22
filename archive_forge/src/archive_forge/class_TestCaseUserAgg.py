import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
class TestCaseUserAgg:

    def __init__(self, user, realm, factory, address=address.IPv4Address('TCP', '127.0.0.1', 54321)):
        self.user = user
        self.transport = proto_helpers.StringTransportWithDisconnection()
        self.protocol = factory.buildProtocol(address)
        self.transport.protocol = self.protocol
        self.user.mind = self.protocol
        self.protocol.makeConnection(self.transport)

    def write(self, stuff):
        self.protocol.dataReceived(stuff)