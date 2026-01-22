import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
class Via:
    """
    A L{Via} is a SIP Via header, representing a segment of the path taken by
    the request.

    See RFC 3261, sections 8.1.1.7, 18.2.2, and 20.42.

    @ivar transport: Network protocol used for this leg. (Probably either "TCP"
    or "UDP".)
    @type transport: C{str}
    @ivar branch: Unique identifier for this request.
    @type branch: C{str}
    @ivar host: Hostname or IP for this leg.
    @type host: C{str}
    @ivar port: Port used for this leg.
    @type port C{int}, or None.
    @ivar rportRequested: Whether to request RFC 3581 client processing or not.
    @type rportRequested: C{bool}
    @ivar rportValue: Servers wishing to honor requests for RFC 3581 processing
    should set this parameter to the source port the request was received
    from.
    @type rportValue: C{int}, or None.

    @ivar ttl: Time-to-live for requests on multicast paths.
    @type ttl: C{int}, or None.
    @ivar maddr: The destination multicast address, if any.
    @type maddr: C{str}, or None.
    @ivar hidden: Obsolete in SIP 2.0.
    @type hidden: C{bool}
    @ivar otherParams: Any other parameters in the header.
    @type otherParams: C{dict}
    """

    def __init__(self, host, port=PORT, transport='UDP', ttl=None, hidden=False, received=None, rport=_absent, branch=None, maddr=None, **kw):
        """
        Set parameters of this Via header. All arguments correspond to
        attributes of the same name.

        To maintain compatibility with old SIP
        code, the 'rport' argument is used to determine the values of
        C{rportRequested} and C{rportValue}. If None, C{rportRequested} is set
        to True. (The deprecated method for doing this is to pass True.) If an
        integer, C{rportValue} is set to the given value.

        Any arguments not explicitly named here are collected into the
        C{otherParams} dict.
        """
        self.transport = transport
        self.host = host
        self.port = port
        self.ttl = ttl
        self.hidden = hidden
        self.received = received
        if rport is True:
            warnings.warn('rport=True is deprecated since Twisted 9.0.', DeprecationWarning, stacklevel=2)
            self.rportValue = None
            self.rportRequested = True
        elif rport is None:
            self.rportValue = None
            self.rportRequested = True
        elif rport is _absent:
            self.rportValue = None
            self.rportRequested = False
        else:
            self.rportValue = rport
            self.rportRequested = False
        self.branch = branch
        self.maddr = maddr
        self.otherParams = kw

    @property
    def rport(self):
        """
        Returns the rport value expected by the old SIP code.
        """
        if self.rportRequested == True:
            return True
        elif self.rportValue is not None:
            return self.rportValue
        else:
            return None

    @rport.setter
    def rport(self, newRPort):
        """
        L{Base._fixupNAT} sets C{rport} directly, so this method sets
        C{rportValue} based on that.

        @param newRPort: The new rport value.
        @type newRPort: C{int}
        """
        self.rportValue = newRPort
        self.rportRequested = False

    def toString(self):
        """
        Serialize this header for use in a request or response.
        """
        s = f'SIP/2.0/{self.transport} {self.host}:{self.port}'
        if self.hidden:
            s += ';hidden'
        for n in ('ttl', 'branch', 'maddr', 'received'):
            value = getattr(self, n)
            if value is not None:
                s += f';{n}={value}'
        if self.rportRequested:
            s += ';rport'
        elif self.rportValue is not None:
            s += f';rport={self.rport}'
        etc = sorted(self.otherParams.items())
        for k, v in etc:
            if v is None:
                s += ';' + k
            else:
                s += f';{k}={v}'
        return s