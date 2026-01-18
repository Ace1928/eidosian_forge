import dns.name
import dns.rdataset
import dns.rdataclass
import dns.renderer
from ._compat import string_types
Convert an RRset into an Rdataset.

        Returns a ``dns.rdataset.Rdataset``.
        