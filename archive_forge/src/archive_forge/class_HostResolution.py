from socket import (
from typing import (
from zope.interface import implementer
from twisted.internet._idna import _idnaBytes
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.internet.threads import deferToThreadPool
from twisted.logger import Logger
from twisted.python.compat import nativeString
@implementer(IHostResolution)
class HostResolution:
    """
    The in-progress resolution of a given hostname.
    """

    def __init__(self, name: str):
        """
        Create a L{HostResolution} with the given name.
        """
        self.name = name

    def cancel(self) -> NoReturn:
        raise NotImplementedError()