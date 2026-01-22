import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def addRecord(self, owner, ttl, type, domain, cls, rdata):
    """
        Add a record to our authority.  Expand domain with origin if necessary.

        @param owner: origin?
        @type owner: L{bytes}

        @param ttl: time to live for the record
        @type ttl: L{int}

        @param domain: the domain for which the record is to be added
        @type domain: L{bytes}

        @param type: record type
        @type type: L{str}

        @param cls: record class
        @type cls: L{str}

        @param rdata: record data
        @type rdata: L{list} of L{bytes}
        """
    if not domain.endswith(b'.'):
        domain = domain + b'.' + owner[:-1]
    else:
        domain = domain[:-1]
    f = getattr(self, f'class_{cls}', None)
    if f:
        f(ttl, type, domain, rdata)
    else:
        raise NotImplementedError(f'Record class {cls!r} not supported')