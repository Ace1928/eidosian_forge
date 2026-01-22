import socket
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.logger import Logger
from twisted.names import dns
from twisted.names.error import (
@implementer(interfaces.IResolver)
class ResolverBase:
    """
    L{ResolverBase} is a base class for implementations of
    L{interfaces.IResolver} which deals with a lot
    of the boilerplate of implementing all of the lookup methods.

    @cvar _errormap: A C{dict} mapping DNS protocol failure response codes
        to exception classes which will be used to represent those failures.
    """
    _log = Logger()
    _errormap = {dns.EFORMAT: DNSFormatError, dns.ESERVER: DNSServerError, dns.ENAME: DNSNameError, dns.ENOTIMP: DNSNotImplementedError, dns.EREFUSED: DNSQueryRefusedError}
    typeToMethod = None

    def __init__(self):
        self.typeToMethod = {}
        for k, v in typeToMethod.items():
            self.typeToMethod[k] = getattr(self, v)

    def exceptionForCode(self, responseCode):
        """
        Convert a response code (one of the possible values of
        L{dns.Message.rCode} to an exception instance representing it.

        @since: 10.0
        """
        return self._errormap.get(responseCode, DNSUnknownError)

    def query(self, query, timeout=None):
        try:
            method = self.typeToMethod[query.type]
        except KeyError:
            self._log.debug('Query of unknown type {query.type} for {query.name.name!r}', query=query)
            return defer.maybeDeferred(self._lookup, query.name.name, dns.IN, query.type, timeout)
        else:
            return defer.maybeDeferred(method, query.name.name, timeout)

    def _lookup(self, name, cls, type, timeout):
        return defer.fail(NotImplementedError('ResolverBase._lookup'))

    def lookupAddress(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.A, timeout)

    def lookupIPV6Address(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.AAAA, timeout)

    def lookupAddress6(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.A6, timeout)

    def lookupMailExchange(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.MX, timeout)

    def lookupNameservers(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.NS, timeout)

    def lookupCanonicalName(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.CNAME, timeout)

    def lookupMailBox(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.MB, timeout)

    def lookupMailGroup(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.MG, timeout)

    def lookupMailRename(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.MR, timeout)

    def lookupPointer(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.PTR, timeout)

    def lookupAuthority(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.SOA, timeout)

    def lookupNull(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.NULL, timeout)

    def lookupWellKnownServices(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.WKS, timeout)

    def lookupService(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.SRV, timeout)

    def lookupHostInfo(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.HINFO, timeout)

    def lookupMailboxInfo(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.MINFO, timeout)

    def lookupText(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.TXT, timeout)

    def lookupSenderPolicy(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.SPF, timeout)

    def lookupResponsibility(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.RP, timeout)

    def lookupAFSDatabase(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.AFSDB, timeout)

    def lookupZone(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.AXFR, timeout)

    def lookupNamingAuthorityPointer(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.NAPTR, timeout)

    def lookupAllRecords(self, name, timeout=None):
        return self._lookup(dns.domainString(name), dns.IN, dns.ALL_RECORDS, timeout)

    def getHostByName(self, name, timeout=None, effort=10):
        name = dns.domainString(name)
        d = self.lookupAllRecords(name, timeout)
        d.addCallback(self._cbRecords, name, effort)
        return d

    def _cbRecords(self, records, name, effort):
        ans, auth, add = records
        result = extractRecord(self, dns.Name(name), ans + auth + add, effort)
        if not result:
            raise error.DNSLookupError(name)
        return result