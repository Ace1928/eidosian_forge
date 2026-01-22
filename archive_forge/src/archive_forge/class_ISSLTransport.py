from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class ISSLTransport(ITCPTransport):
    """
    A SSL/TLS based transport.
    """

    def getPeerCertificate() -> object:
        """
        Return an object with the peer's certificate info.
        """