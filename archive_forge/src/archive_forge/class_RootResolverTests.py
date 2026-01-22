from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import (
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase
class RootResolverTests(TestCase):
    """
    Tests for L{twisted.names.root.Resolver}.
    """

    def _queryTest(self, filter):
        """
        Invoke L{Resolver._query} and verify that it sends the correct DNS
        query.  Deliver a canned response to the query and return whatever the
        L{Deferred} returned by L{Resolver._query} fires with.

        @param filter: The value to pass for the C{filter} parameter to
            L{Resolver._query}.
        """
        reactor = MemoryReactor()
        resolver = Resolver([], reactor=reactor)
        d = resolver._query(Query(b'foo.example.com', A, IN), [('1.1.2.3', 1053)], (30,), filter)
        portNumber, transport = reactor.udpPorts.popitem()
        [(packet, address)] = transport._sentPackets
        message = Message()
        message.fromStr(packet)
        self.assertEqual(message.queries, [Query(b'foo.example.com', A, IN)])
        self.assertEqual(message.answers, [])
        self.assertEqual(message.authority, [])
        self.assertEqual(message.additional, [])
        response = []
        d.addCallback(response.append)
        self.assertEqual(response, [])
        del message.queries[:]
        message.answer = 1
        message.answers.append(RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21')))
        transport._protocol.datagramReceived(message.toStr(), ('1.1.2.3', 1053))
        return response[0]

    def test_filteredQuery(self):
        """
        L{Resolver._query} accepts a L{Query} instance and an address, issues
        the query, and returns a L{Deferred} which fires with the response to
        the query.  If a true value is passed for the C{filter} parameter, the
        result is a three-tuple of lists of records.
        """
        answer, authority, additional = self._queryTest(True)
        self.assertEqual(answer, [RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21', ttl=0))])
        self.assertEqual(authority, [])
        self.assertEqual(additional, [])

    def test_unfilteredQuery(self):
        """
        Similar to L{test_filteredQuery}, but for the case where a false value
        is passed for the C{filter} parameter.  In this case, the result is a
        L{Message} instance.
        """
        message = self._queryTest(False)
        self.assertIsInstance(message, Message)
        self.assertEqual(message.queries, [])
        self.assertEqual(message.answers, [RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21', ttl=0))])
        self.assertEqual(message.authority, [])
        self.assertEqual(message.additional, [])

    def _respond(self, answers=[], authority=[], additional=[], rCode=OK):
        """
        Create a L{Message} suitable for use as a response to a query.

        @param answers: A C{list} of two-tuples giving data for the answers
            section of the message.  The first element of each tuple is a name
            for the L{RRHeader}.  The second element is the payload.
        @param authority: A C{list} like C{answers}, but for the authority
            section of the response.
        @param additional: A C{list} like C{answers}, but for the
            additional section of the response.
        @param rCode: The response code the message will be created with.

        @return: A new L{Message} initialized with the given values.
        """
        response = Message(rCode=rCode)
        for section, data in [(response.answers, answers), (response.authority, authority), (response.additional, additional)]:
            section.extend([RRHeader(name, record.TYPE, getattr(record, 'CLASS', IN), payload=record) for name, record in data])
        return response

    def _getResolver(self, serverResponses, maximumQueries=10):
        """
        Create and return a new L{root.Resolver} modified to resolve queries
        against the record data represented by C{servers}.

        @param serverResponses: A mapping from dns server addresses to
            mappings.  The inner mappings are from query two-tuples (name,
            type) to dictionaries suitable for use as **arguments to
            L{_respond}.  See that method for details.
        """
        roots = ['1.1.2.3']
        resolver = Resolver(roots, maximumQueries)

        def query(query, serverAddresses, timeout, filter):
            msg(f'Query for QNAME {query.name} at {serverAddresses!r}')
            for addr in serverAddresses:
                try:
                    server = serverResponses[addr]
                except KeyError:
                    continue
                records = server[query.name.name, query.type]
                return succeed(self._respond(**records))
        resolver._query = query
        return resolver

    def test_lookupAddress(self):
        """
        L{root.Resolver.lookupAddress} looks up the I{A} records for the
        specified hostname by first querying one of the root servers the
        resolver was created with and then following the authority delegations
        until a result is received.
        """
        servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'authority': [(b'foo.example.com', Record_NS(b'ns1.example.com'))], 'additional': [(b'ns1.example.com', Record_A('34.55.89.144'))]}}, ('34.55.89.144', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', Record_A('10.0.0.1'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'foo.example.com')
        d.addCallback(getOneAddress)
        d.addCallback(self.assertEqual, '10.0.0.1')
        return d

    def test_lookupChecksClass(self):
        """
        If a response includes a record with a class different from the one
        in the query, it is ignored and lookup continues until a record with
        the right class is found.
        """
        badClass = Record_A('10.0.0.1')
        badClass.CLASS = HS
        servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', badClass)], 'authority': [(b'foo.example.com', Record_NS(b'ns1.example.com'))], 'additional': [(b'ns1.example.com', Record_A('10.0.0.2'))]}}, ('10.0.0.2', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', Record_A('10.0.0.3'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'foo.example.com')
        d.addCallback(getOnePayload)
        d.addCallback(self.assertEqual, Record_A('10.0.0.3'))
        return d

    def test_missingGlue(self):
        """
        If an intermediate response includes no glue records for the
        authorities, separate queries are made to find those addresses.
        """
        servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'authority': [(b'foo.example.com', Record_NS(b'ns1.example.org'))]}, (b'ns1.example.org', A): {'answers': [(b'ns1.example.org', Record_A('10.0.0.1'))]}}, ('10.0.0.1', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', Record_A('10.0.0.2'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'foo.example.com')
        d.addCallback(getOneAddress)
        d.addCallback(self.assertEqual, '10.0.0.2')
        return d

    def test_missingName(self):
        """
        If a name is missing, L{Resolver.lookupAddress} returns a L{Deferred}
        which fails with L{DNSNameError}.
        """
        servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'rCode': ENAME}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'foo.example.com')
        return self.assertFailure(d, DNSNameError)

    def test_answerless(self):
        """
        If a query is responded to with no answers or nameserver records, the
        L{Deferred} returned by L{Resolver.lookupAddress} fires with
        L{ResolverError}.
        """
        servers = {('1.1.2.3', 53): {(b'example.com', A): {}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        return self.assertFailure(d, ResolverError)

    def test_delegationLookupError(self):
        """
        If there is an error resolving the nameserver in a delegation response,
        the L{Deferred} returned by L{Resolver.lookupAddress} fires with that
        error.
        """
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns1.example.com'))]}, (b'ns1.example.com', A): {'rCode': ENAME}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        return self.assertFailure(d, DNSNameError)

    def test_delegationLookupEmpty(self):
        """
        If there are no records in the response to a lookup of a delegation
        nameserver, the L{Deferred} returned by L{Resolver.lookupAddress} fires
        with L{ResolverError}.
        """
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns1.example.com'))]}, (b'ns1.example.com', A): {}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        return self.assertFailure(d, ResolverError)

    def test_lookupNameservers(self):
        """
        L{Resolver.lookupNameservers} is like L{Resolver.lookupAddress}, except
        it queries for I{NS} records instead of I{A} records.
        """
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'rCode': ENAME}, (b'example.com', NS): {'answers': [(b'example.com', Record_NS(b'ns1.example.com'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupNameservers(b'example.com')

        def getOneName(results):
            ans, auth, add = results
            return ans[0].payload.name
        d.addCallback(getOneName)
        d.addCallback(self.assertEqual, Name(b'ns1.example.com'))
        return d

    def test_returnCanonicalName(self):
        """
        If a I{CNAME} record is encountered as the answer to a query for
        another record type, that record is returned as the answer.
        """
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_CNAME(b'example.net')), (b'example.net', Record_A('10.0.0.7'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        d.addCallback(lambda results: results[0])
        d.addCallback(self.assertEqual, [RRHeader(b'example.com', CNAME, payload=Record_CNAME(b'example.net')), RRHeader(b'example.net', A, payload=Record_A('10.0.0.7'))])
        return d

    def test_followCanonicalName(self):
        """
        If no record of the requested type is included in a response, but a
        I{CNAME} record for the query name is included, queries are made to
        resolve the value of the I{CNAME}.
        """
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_CNAME(b'example.net'))]}, (b'example.net', A): {'answers': [(b'example.net', Record_A('10.0.0.5'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        d.addCallback(lambda results: results[0])
        d.addCallback(self.assertEqual, [RRHeader(b'example.com', CNAME, payload=Record_CNAME(b'example.net')), RRHeader(b'example.net', A, payload=Record_A('10.0.0.5'))])
        return d

    def test_detectCanonicalNameLoop(self):
        """
        If there is a cycle between I{CNAME} records in a response, this is
        detected and the L{Deferred} returned by the lookup method fails
        with L{ResolverError}.
        """
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_CNAME(b'example.net')), (b'example.net', Record_CNAME(b'example.com'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        return self.assertFailure(d, ResolverError)

    def test_boundedQueries(self):
        """
        L{Resolver.lookupAddress} won't issue more queries following
        delegations than the limit passed to its initializer.
        """
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns1.example.com'))]}, (b'ns1.example.com', A): {'answers': [(b'ns1.example.com', Record_A('10.0.0.2'))]}}, ('10.0.0.2', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns2.example.com'))], 'additional': [(b'ns2.example.com', Record_A('10.0.0.3'))]}}, ('10.0.0.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_A('10.0.0.4'))]}}}
        failer = self._getResolver(servers, 3)
        failD = self.assertFailure(failer.lookupAddress(b'example.com'), ResolverError)
        succeeder = self._getResolver(servers, 4)
        succeedD = succeeder.lookupAddress(b'example.com')
        succeedD.addCallback(getOnePayload)
        succeedD.addCallback(self.assertEqual, Record_A('10.0.0.4'))
        return gatherResults([failD, succeedD])