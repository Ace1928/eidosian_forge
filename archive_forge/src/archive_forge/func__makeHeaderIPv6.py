from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def _makeHeaderIPv6(sig: bytes=V2_SIGNATURE, verCom: bytes=b'!', famProto: bytes=b'!', addrLength: bytes=b'\x00$', addrs: bytes=(b'\x00' * 15 + b'\x01') * 2, ports: bytes=b'\x1f\x90"\xb8') -> bytes:
    """
    Construct a version 2 IPv6 header with custom bytes.

    @param sig: The protocol signature; defaults to valid L{V2_SIGNATURE}.
    @type sig: L{bytes}

    @param verCom: Protocol version and command.  Defaults to V2 PROXY.
    @type verCom: L{bytes}

    @param famProto: Address family and protocol.  Defaults to AF_INET6/STREAM.
    @type famProto: L{bytes}

    @param addrLength: Network-endian byte length of payload.  Defaults to
        description of default addrs/ports.
    @type addrLength: L{bytes}

    @param addrs: Address payload.  Defaults to C{::1} for source and
        destination.
    @type addrs: L{bytes}

    @param ports: Source and destination ports.  Defaults to 8080 for source
        8888 for destination.
    @type ports: L{bytes}

    @return: A packet with header, addresses, and ports.
    @rtype: L{bytes}
    """
    return sig + verCom + famProto + addrLength + addrs + ports