from typing import Tuple, Union
from zope.interface import implementer
from twisted.internet import address
from . import _info, _interfaces
from ._exceptions import (

        Parse a bytestring as a full PROXY protocol header line.

        @param line: A bytestring that represents a valid HAProxy PROXY
            protocol header line.
        @type line: bytes

        @return: A L{_interfaces.IProxyInfo} containing the parsed data.

        @raises InvalidProxyHeader: If the bytestring does not represent a
            valid PROXY header.

        @raises InvalidNetworkProtocol: When no protocol can be parsed or is
            not one of the allowed values.

        @raises MissingAddressData: When the protocol is TCP* but the header
            does not contain a complete set of addresses and ports.
        