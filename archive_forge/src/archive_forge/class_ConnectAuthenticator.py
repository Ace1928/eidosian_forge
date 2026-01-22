from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import (
class ConnectAuthenticator(Authenticator):
    """
    Authenticator for initiating entities.
    """
    namespace: Optional[str] = None

    def __init__(self, otherHost):
        self.otherHost = otherHost

    def connectionMade(self):
        self.xmlstream.namespace = self.namespace
        self.xmlstream.otherEntity = jid.internJID(self.otherHost)
        self.xmlstream.sendHeader()

    def initializeStream(self):
        """
        Perform stream initialization procedures.

        An L{XmlStream} holds a list of initializer objects in its
        C{initializers} attribute. This method calls these initializers in
        order and dispatches the L{STREAM_AUTHD_EVENT} event when the list has
        been successfully processed. Otherwise it dispatches the
        C{INIT_FAILED_EVENT} event with the failure.

        Initializers may return the special L{Reset} object to halt the
        initialization processing. It signals that the current initializer was
        successfully processed, but that the XML Stream has been reset. An
        example is the TLSInitiatingInitializer.
        """

        def remove_first(result):
            self.xmlstream.initializers.pop(0)
            return result

        def do_next(result):
            """
            Take the first initializer and process it.

            On success, the initializer is removed from the list and
            then next initializer will be tried.
            """
            if result is Reset:
                return None
            try:
                init = self.xmlstream.initializers[0]
            except IndexError:
                self.xmlstream.dispatch(self.xmlstream, STREAM_AUTHD_EVENT)
                return None
            else:
                d = defer.maybeDeferred(init.initialize)
                d.addCallback(remove_first)
                d.addCallback(do_next)
                return d
        d = defer.succeed(None)
        d.addCallback(do_next)
        d.addErrback(self.xmlstream.dispatch, INIT_FAILED_EVENT)

    def streamStarted(self, rootElement):
        """
        Called by the XmlStream when the stream has started.

        This extends L{Authenticator.streamStarted} to extract further stream
        headers from C{rootElement}, optionally wait for stream features being
        received and then call C{initializeStream}.
        """
        Authenticator.streamStarted(self, rootElement)
        self.xmlstream.sid = rootElement.getAttribute('id')
        if rootElement.hasAttribute('from'):
            self.xmlstream.otherEntity = jid.internJID(rootElement['from'])
        if self.xmlstream.version >= (1, 0):

            def onFeatures(element):
                features = {}
                for feature in element.elements():
                    features[feature.uri, feature.name] = feature
                self.xmlstream.features = features
                self.initializeStream()
            self.xmlstream.addOnetimeObserver('/features[@xmlns="%s"]' % NS_STREAMS, onFeatures)
        else:
            self.initializeStream()