import socket
import dns.ipv4
import dns.ipv6
from ._compat import maybe_ord
Is the textual-form network address a multicast address?

    *text*, a ``text``, the textual address.

    Raises ``ValueError`` if the address family cannot be determined
    from the input.

    Returns a ``bool``.
    