from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.names import common, dns, error
@implementer(interfaces.IResolver)
class ResolverChain(common.ResolverBase):
    """
    Lookup an address using multiple L{IResolver}s
    """

    def __init__(self, resolvers):
        """
        @type resolvers: L{list}
        @param resolvers: A L{list} of L{IResolver} providers.
        """
        common.ResolverBase.__init__(self)
        self.resolvers = resolvers

    def _lookup(self, name, cls, type, timeout):
        """
        Build a L{dns.Query} for the given parameters and dispatch it
        to each L{IResolver} in C{self.resolvers} until an answer or
        L{error.AuthoritativeDomainError} is returned.

        @type name: C{str}
        @param name: DNS name to resolve.

        @type type: C{int}
        @param type: DNS record type.

        @type cls: C{int}
        @param cls: DNS record class.

        @type timeout: Sequence of C{int}
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @rtype: L{Deferred}
        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """
        if not self.resolvers:
            return defer.fail(error.DomainError())
        q = dns.Query(name, type, cls)
        d = self.resolvers[0].query(q, timeout)
        for r in self.resolvers[1:]:
            d = d.addErrback(FailureHandler(r.query, q, timeout))
        return d

    def lookupAllRecords(self, name, timeout=None):
        if not self.resolvers:
            return defer.fail(error.DomainError())
        d = self.resolvers[0].lookupAllRecords(name, timeout)
        for r in self.resolvers[1:]:
            d = d.addErrback(FailureHandler(r.lookupAllRecords, name, timeout))
        return d