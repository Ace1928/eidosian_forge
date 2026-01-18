from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
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