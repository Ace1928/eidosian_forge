import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials

            Represents the state of an option on side of the telnet
            connection.  Some options can be enabled on a particular side of
            the connection (RFC 1073 for example: only the client can have
            NAWS enabled).  Other options can be enabled on either or both
            sides (such as RFC 1372: each side can have its own flow control
            state).

            @ivar state: C{'yes'} or C{'no'} indicating whether or not this
                option is enabled on one side of the connection.

            @ivar negotiating: A boolean tracking whether negotiation about
                this option is in progress.

            @ivar onResult: When negotiation about this option has been
                initiated by this side of the connection, a L{Deferred}
                which will fire with the result of the negotiation.  L{None}
                at other times.
            