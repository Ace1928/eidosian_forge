from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
class InitiatingInitializerHarness:
    """
    Testing harness for interacting with XML stream initializers.

    This sets up an L{utility.XmlPipe} to create a communication channel between
    the initializer and the stubbed receiving entity. It features a sink and
    source side that both act similarly to a real L{xmlstream.XmlStream}. The
    sink is augmented with an authenticator to which initializers can be added.

    The harness also provides some utility methods to work with event observers
    and deferreds.
    """

    def setUp(self):
        self.output = []
        self.pipe = utility.XmlPipe()
        self.xmlstream = self.pipe.sink
        self.authenticator = xmlstream.ConnectAuthenticator('example.org')
        self.xmlstream.authenticator = self.authenticator

    def waitFor(self, event, handler):
        """
        Observe an output event, returning a deferred.

        The returned deferred will be fired when the given event has been
        observed on the source end of the L{XmlPipe} tied to the protocol
        under test. The handler is added as the first callback.

        @param event: The event to be observed. See
            L{utility.EventDispatcher.addOnetimeObserver}.
        @param handler: The handler to be called with the observed event object.
        @rtype: L{defer.Deferred}.
        """
        d = defer.Deferred()
        d.addCallback(handler)
        self.pipe.source.addOnetimeObserver(event, d.callback)
        return d