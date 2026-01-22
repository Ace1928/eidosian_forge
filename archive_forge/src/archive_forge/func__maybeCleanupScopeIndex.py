import socket
import sys
from ctypes import (
from ctypes.util import find_library
from socket import AF_INET, AF_INET6, inet_ntop
from typing import Any, List, Tuple
from twisted.python.compat import nativeString
def _maybeCleanupScopeIndex(family, packed):
    """
    On FreeBSD, kill the embedded interface indices in link-local scoped
    addresses.

    @param family: The address family of the packed address - one of the
        I{socket.AF_*} constants.

    @param packed: The packed representation of the address (ie, the bytes of a
        I{in_addr} field).
    @type packed: L{bytes}

    @return: The packed address with any FreeBSD-specific extra bits cleared.
    @rtype: L{bytes}

    @see: U{https://twistedmatrix.com/trac/ticket/6843}
    @see: U{http://www.freebsd.org/doc/en/books/developers-handbook/ipv6.html#ipv6-scope-index}

    @note: Indications are that the need for this will be gone in FreeBSD >=10.
    """
    if sys.platform.startswith('freebsd') and packed[:2] == b'\xfe\x80':
        return packed[:2] + b'\x00\x00' + packed[4:]
    return packed