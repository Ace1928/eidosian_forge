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
@implementer(ijabber.IInitiatingInitializer)
class BaseFeatureInitiatingInitializer:
    """
    Base class for initializers with a stream feature.

    This assumes the associated XmlStream represents the initiating entity
    of the connection.

    @cvar feature: tuple of (uri, name) of the stream feature root element.
    @type feature: tuple of (C{str}, C{str})

    @ivar required: whether the stream feature is required to be advertized
                    by the receiving entity.
    @type required: C{bool}
    """
    feature: Optional[Tuple[str, str]] = None

    def __init__(self, xs, required=False):
        self.xmlstream = xs
        self.required = required

    def initialize(self):
        """
        Initiate the initialization.

        Checks if the receiving entity advertizes the stream feature. If it
        does, the initialization is started. If it is not advertized, and the
        C{required} instance variable is C{True}, it raises
        L{FeatureNotAdvertized}. Otherwise, the initialization silently
        succeeds.
        """
        if self.feature in self.xmlstream.features:
            return self.start()
        elif self.required:
            raise FeatureNotAdvertized
        else:
            return None

    def start(self):
        """
        Start the actual initialization.

        May return a deferred for asynchronous initialization.
        """