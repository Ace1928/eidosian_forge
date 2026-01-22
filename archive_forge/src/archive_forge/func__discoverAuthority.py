from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
def _discoverAuthority(self, query, servers, timeout, queriesLeft):
    """
        Issue a query to a server and follow a delegation if necessary.

        @param query: The query to issue.
        @type query: L{dns.Query}

        @param servers: The servers which might have an answer for this
            query.
        @type servers: L{list} of L{tuple} of L{str} and L{int}

        @param timeout: A C{tuple} of C{int} giving the timeout to use for this
            query.

        @param queriesLeft: A C{int} giving the number of queries which may
            yet be attempted to answer this query before the attempt will be
            abandoned.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} giving the response, or with a
            L{Failure} if there is a timeout or response error.
        """
    if queriesLeft <= 0:
        return Failure(error.ResolverError('Query limit reached without result'))
    d = self._query(query, servers, timeout, False)
    d.addCallback(self._discoveredAuthority, query, timeout, queriesLeft - 1)
    return d