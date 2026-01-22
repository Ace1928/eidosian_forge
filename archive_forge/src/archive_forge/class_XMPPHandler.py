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
@implementer(ijabber.IXMPPHandler)
class XMPPHandler:
    """
    XMPP protocol handler.

    Classes derived from this class implement (part of) one or more XMPP
    extension protocols, and are referred to as a subprotocol implementation.
    """

    def __init__(self):
        self.parent = None
        self.xmlstream = None

    def setHandlerParent(self, parent):
        self.parent = parent
        self.parent.addHandler(self)

    def disownHandlerParent(self, parent):
        self.parent.removeHandler(self)
        self.parent = None

    def makeConnection(self, xs):
        self.xmlstream = xs
        self.connectionMade()

    def connectionMade(self):
        """
        Called after a connection has been established.

        Can be overridden to perform work before stream initialization.
        """

    def connectionInitialized(self):
        """
        The XML stream has been initialized.

        Can be overridden to perform work after stream initialization, e.g. to
        set up observers and start exchanging XML stanzas.
        """

    def connectionLost(self, reason):
        """
        The XML stream has been closed.

        This method can be extended to inspect the C{reason} argument and
        act on it.
        """
        self.xmlstream = None

    def send(self, obj):
        """
        Send data over the managed XML stream.

        @note: The stream manager maintains a queue for data sent using this
               method when there is no current initialized XML stream. This
               data is then sent as soon as a new stream has been established
               and initialized. Subsequently, L{connectionInitialized} will be
               called again. If this queueing is not desired, use C{send} on
               C{self.xmlstream}.

        @param obj: data to be sent over the XML stream. This is usually an
                    object providing L{domish.IElement}, or serialized XML. See
                    L{xmlstream.XmlStream} for details.
        """
        self.parent.send(obj)