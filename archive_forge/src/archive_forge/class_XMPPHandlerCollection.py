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
@implementer(ijabber.IXMPPHandlerCollection)
class XMPPHandlerCollection:
    """
    Collection of XMPP subprotocol handlers.

    This allows for grouping of subprotocol handlers, but is not an
    L{XMPPHandler} itself, so this is not recursive.

    @ivar handlers: List of protocol handlers.
    @type handlers: C{list} of objects providing
                      L{IXMPPHandler}
    """

    def __init__(self):
        self.handlers = []

    def __iter__(self):
        """
        Act as a container for handlers.
        """
        return iter(self.handlers)

    def addHandler(self, handler):
        """
        Add protocol handler.

        Protocol handlers are expected to provide L{ijabber.IXMPPHandler}.
        """
        self.handlers.append(handler)

    def removeHandler(self, handler):
        """
        Remove protocol handler.
        """
        self.handlers.remove(handler)