import dns.exception
import dns.name
import dns.resolver
from ._compat import string_types, maybe_decode
Look for NAPTR RRs for the specified number in the specified domains.

    e.g. lookup('16505551212', ['e164.dnspython.org.', 'e164.arpa.'])

    *number*, a ``text`` is the number to look for.

    *domains* is an iterable containing ``dns.name.Name`` values.

    *resolver*, a ``dns.resolver.Resolver``, is the resolver to use.  If
    ``None``, the default resolver is used.
    