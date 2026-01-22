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
class XmlStream(xmlstream.XmlStream):
    """
    XMPP XML Stream protocol handler.

    @ivar version: XML stream version as a tuple (major, minor). Initially,
                   this is set to the minimally supported version. Upon
                   receiving the stream header of the peer, it is set to the
                   minimum of that value and the version on the received
                   header.
    @type version: (C{int}, C{int})
    @ivar namespace: default namespace URI for stream
    @type namespace: C{unicode}
    @ivar thisEntity: JID of this entity
    @type thisEntity: L{JID}
    @ivar otherEntity: JID of the peer entity
    @type otherEntity: L{JID}
    @ivar sid: session identifier
    @type sid: C{unicode}
    @ivar initiating: True if this is the initiating stream
    @type initiating: C{bool}
    @ivar features: map of (uri, name) to stream features element received from
                    the receiving entity.
    @type features: C{dict} of (C{unicode}, C{unicode}) to L{domish.Element}.
    @ivar prefixes: map of URI to prefixes that are to appear on stream
                    header.
    @type prefixes: C{dict} of C{unicode} to C{unicode}
    @ivar initializers: list of stream initializer objects
    @type initializers: C{list} of objects that provide L{IInitializer}
    @ivar authenticator: associated authenticator that uses C{initializers} to
                         initialize the XML stream.
    """
    version = (1, 0)
    namespace = 'invalid'
    thisEntity = None
    otherEntity = None
    sid = None
    initiating = True
    _headerSent = False

    def __init__(self, authenticator):
        xmlstream.XmlStream.__init__(self)
        self.prefixes = {NS_STREAMS: 'stream'}
        self.authenticator = authenticator
        self.initializers = []
        self.features = {}
        authenticator.associateWithStream(self)

    def _callLater(self, *args, **kwargs):
        from twisted.internet import reactor
        return reactor.callLater(*args, **kwargs)

    def reset(self):
        """
        Reset XML Stream.

        Resets the XML Parser for incoming data. This is to be used after
        successfully negotiating a new layer, e.g. TLS and SASL. Note that
        registered event observers will continue to be in place.
        """
        self._headerSent = False
        self._initializeStream()

    def onStreamError(self, errelem):
        """
        Called when a stream:error element has been received.

        Dispatches a L{STREAM_ERROR_EVENT} event with the error element to
        allow for cleanup actions and drops the connection.

        @param errelem: The received error element.
        @type errelem: L{domish.Element}
        """
        self.dispatch(failure.Failure(error.exceptionFromStreamError(errelem)), STREAM_ERROR_EVENT)
        self.transport.loseConnection()

    def sendHeader(self):
        """
        Send stream header.
        """
        localPrefixes = {}
        for uri, prefix in self.prefixes.items():
            if uri != NS_STREAMS:
                localPrefixes[prefix] = uri
        rootElement = domish.Element((NS_STREAMS, 'stream'), self.namespace, localPrefixes=localPrefixes)
        if self.otherEntity:
            rootElement['to'] = self.otherEntity.userhost()
        if self.thisEntity:
            rootElement['from'] = self.thisEntity.userhost()
        if not self.initiating and self.sid:
            rootElement['id'] = self.sid
        if self.version >= (1, 0):
            rootElement['version'] = '%d.%d' % self.version
        self.send(rootElement.toXml(prefixes=self.prefixes, closeElement=0))
        self._headerSent = True

    def sendFooter(self):
        """
        Send stream footer.
        """
        self.send('</stream:stream>')

    def sendStreamError(self, streamError):
        """
        Send stream level error.

        If we are the receiving entity, and haven't sent the header yet,
        we sent one first.

        After sending the stream error, the stream is closed and the transport
        connection dropped.

        @param streamError: stream error instance
        @type streamError: L{error.StreamError}
        """
        if not self._headerSent and (not self.initiating):
            self.sendHeader()
        if self._headerSent:
            self.send(streamError.getElement())
            self.sendFooter()
        self.transport.loseConnection()

    def send(self, obj):
        """
        Send data over the stream.

        This overrides L{xmlstream.XmlStream.send} to use the default namespace
        of the stream header when serializing L{domish.IElement}s. It is
        assumed that if you pass an object that provides L{domish.IElement},
        it represents a direct child of the stream's root element.
        """
        if domish.IElement.providedBy(obj):
            obj = obj.toXml(prefixes=self.prefixes, defaultUri=self.namespace, prefixesInScope=list(self.prefixes.values()))
        xmlstream.XmlStream.send(self, obj)

    def connectionMade(self):
        """
        Called when a connection is made.

        Notifies the authenticator when a connection has been made.
        """
        xmlstream.XmlStream.connectionMade(self)
        self.authenticator.connectionMade()

    def onDocumentStart(self, rootElement):
        """
        Called when the stream header has been received.

        Extracts the header's C{id} and C{version} attributes from the root
        element. The C{id} attribute is stored in our C{sid} attribute and the
        C{version} attribute is parsed and the minimum of the version we sent
        and the parsed C{version} attribute is stored as a tuple (major, minor)
        in this class' C{version} attribute. If no C{version} attribute was
        present, we assume version 0.0.

        If appropriate (we are the initiating stream and the minimum of our and
        the other party's version is at least 1.0), a one-time observer is
        registered for getting the stream features. The registered function is
        C{onFeatures}.

        Ultimately, the authenticator's C{streamStarted} method will be called.

        @param rootElement: The root element.
        @type rootElement: L{domish.Element}
        """
        xmlstream.XmlStream.onDocumentStart(self, rootElement)
        self.addOnetimeObserver("/error[@xmlns='%s']" % NS_STREAMS, self.onStreamError)
        self.authenticator.streamStarted(rootElement)