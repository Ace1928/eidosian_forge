from typing import Optional
from zope.interface import implementer
import attr
from twisted.internet.interfaces import IAddress
from ._interfaces import IProxyInfo

    A data container for parsed PROXY protocol information.

    @ivar header: The raw header bytes extracted from the connection.
    @type header: C{bytes}
    @ivar source: The connection source address.
    @type source: L{twisted.internet.interfaces.IAddress}
    @ivar destination: The connection destination address.
    @type destination: L{twisted.internet.interfaces.IAddress}
    