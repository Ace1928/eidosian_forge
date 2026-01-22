import email.message
import email.parser
import errno
import glob
import io
import os
import pickle
import shutil
import signal
import sys
import tempfile
import textwrap
import time
from hashlib import md5
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.portal
import twisted.mail.alias
import twisted.mail.mail
import twisted.mail.maildir
import twisted.mail.protocols
import twisted.mail.relay
import twisted.mail.relaymanager
from twisted import cred, mail
from twisted.internet import address, defer, interfaces, protocol, reactor, task
from twisted.internet.defer import Deferred
from twisted.internet.error import (
from twisted.internet.testing import (
from twisted.mail import pop3, smtp
from twisted.mail.relaymanager import _AttemptManager
from twisted.names import dns
from twisted.names.dns import Record_CNAME, Record_MX, RRHeader
from twisted.names.error import DNSNameError
from twisted.python import failure, log
from twisted.python.filepath import FilePath
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
from twisted.names import client, common, server
@skipIf(platformType != 'posix', 'twisted.mail only works on posix')
class MXTests(TestCase):
    """
    Tests for L{mail.relaymanager.MXCalculator}.
    """

    def setUp(self):
        setUpDNS(self)
        self.clock = task.Clock()
        self.mx = mail.relaymanager.MXCalculator(self.resolver, self.clock)

    def tearDown(self):
        return tearDownDNS(self)

    def test_defaultClock(self):
        """
        L{MXCalculator}'s default clock is C{twisted.internet.reactor}.
        """
        self.assertIdentical(mail.relaymanager.MXCalculator(self.resolver).clock, reactor)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testSimpleSuccess(self):
        self.auth.addresses['test.domain'] = ['the.email.test.domain']
        return self.mx.getMX('test.domain').addCallback(self._cbSimpleSuccess)

    def _cbSimpleSuccess(self, mx):
        self.assertEqual(mx.preference, 0)
        self.assertEqual(str(mx.name), 'the.email.test.domain')

    def testSimpleFailure(self):
        self.mx.fallbackToDomain = False
        return self.assertFailure(self.mx.getMX('test.domain'), IOError)

    def testSimpleFailureWithFallback(self):
        return self.assertFailure(self.mx.getMX('test.domain'), DNSLookupError)

    def _exchangeTest(self, domain, records, correctMailExchange):
        """
        Issue an MX request for the given domain and arrange for it to be
        responded to with the given records.  Verify that the resulting mail
        exchange is the indicated host.

        @type domain: C{str}
        @type records: C{list} of L{RRHeader}
        @type correctMailExchange: C{str}
        @rtype: L{Deferred}
        """

        class DummyResolver:

            def lookupMailExchange(self, name):
                if name == domain:
                    return defer.succeed((records, [], []))
                return defer.fail(DNSNameError(domain))
        self.mx.resolver = DummyResolver()
        d = self.mx.getMX(domain)

        def gotMailExchange(record):
            self.assertEqual(str(record.name), correctMailExchange)
        d.addCallback(gotMailExchange)
        return d

    def test_mailExchangePreference(self):
        """
        The MX record with the lowest preference is returned by
        L{MXCalculator.getMX}.
        """
        domain = 'example.com'
        good = 'good.example.com'
        bad = 'bad.example.com'
        records = [RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(1, bad)), RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(0, good)), RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(2, bad))]
        return self._exchangeTest(domain, records, good)

    def test_badExchangeExcluded(self):
        """
        L{MXCalculator.getMX} returns the MX record with the lowest preference
        which is not also marked as bad.
        """
        domain = 'example.com'
        good = 'good.example.com'
        bad = 'bad.example.com'
        records = [RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(0, bad)), RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(1, good))]
        self.mx.markBad(bad)
        return self._exchangeTest(domain, records, good)

    def test_fallbackForAllBadExchanges(self):
        """
        L{MXCalculator.getMX} returns the MX record with the lowest preference
        if all the MX records in the response have been marked bad.
        """
        domain = 'example.com'
        bad = 'bad.example.com'
        worse = 'worse.example.com'
        records = [RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(0, bad)), RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(1, worse))]
        self.mx.markBad(bad)
        self.mx.markBad(worse)
        return self._exchangeTest(domain, records, bad)

    def test_badExchangeExpires(self):
        """
        L{MXCalculator.getMX} returns the MX record with the lowest preference
        if it was last marked bad longer than L{MXCalculator.timeOutBadMX}
        seconds ago.
        """
        domain = 'example.com'
        good = 'good.example.com'
        previouslyBad = 'bad.example.com'
        records = [RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(0, previouslyBad)), RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(1, good))]
        self.mx.markBad(previouslyBad)
        self.clock.advance(self.mx.timeOutBadMX)
        return self._exchangeTest(domain, records, previouslyBad)

    def test_goodExchangeUsed(self):
        """
        L{MXCalculator.getMX} returns the MX record with the lowest preference
        if it was marked good after it was marked bad.
        """
        domain = 'example.com'
        good = 'good.example.com'
        previouslyBad = 'bad.example.com'
        records = [RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(0, previouslyBad)), RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(1, good))]
        self.mx.markBad(previouslyBad)
        self.mx.markGood(previouslyBad)
        self.clock.advance(self.mx.timeOutBadMX)
        return self._exchangeTest(domain, records, previouslyBad)

    def test_successWithoutResults(self):
        """
        If an MX lookup succeeds but the result set is empty,
        L{MXCalculator.getMX} should try to look up an I{A} record for the
        requested name and call back its returned Deferred with that
        address.
        """
        ip = '1.2.3.4'
        domain = 'example.org'

        class DummyResolver:
            """
            Fake resolver which will respond to an MX lookup with an empty
            result set.

            @ivar mx: A dictionary mapping hostnames to three-tuples of
                results to be returned from I{MX} lookups.

            @ivar a: A dictionary mapping hostnames to addresses to be
                returned from I{A} lookups.
            """
            mx = {domain: ([], [], [])}
            a = {domain: ip}

            def lookupMailExchange(self, domain):
                return defer.succeed(self.mx[domain])

            def getHostByName(self, domain):
                return defer.succeed(self.a[domain])
        self.mx.resolver = DummyResolver()
        d = self.mx.getMX(domain)
        d.addCallback(self.assertEqual, Record_MX(name=ip))
        return d

    def test_failureWithSuccessfulFallback(self):
        """
        Test that if the MX record lookup fails, fallback is enabled, and an A
        record is available for the name, then the Deferred returned by
        L{MXCalculator.getMX} ultimately fires with a Record_MX instance which
        gives the address in the A record for the name.
        """

        class DummyResolver:
            """
            Fake resolver which will fail an MX lookup but then succeed a
            getHostByName call.
            """

            def lookupMailExchange(self, domain):
                return defer.fail(DNSNameError())

            def getHostByName(self, domain):
                return defer.succeed('1.2.3.4')
        self.mx.resolver = DummyResolver()
        d = self.mx.getMX('domain')
        d.addCallback(self.assertEqual, Record_MX(name='1.2.3.4'))
        return d

    def test_cnameWithoutGlueRecords(self):
        """
        If an MX lookup returns a single CNAME record as a result, MXCalculator
        will perform an MX lookup for the canonical name indicated and return
        the MX record which results.
        """
        alias = 'alias.example.com'
        canonical = 'canonical.example.com'
        exchange = 'mail.example.com'

        class DummyResolver:
            """
            Fake resolver which will return a CNAME for an MX lookup of a name
            which is an alias and an MX for an MX lookup of the canonical name.
            """

            def lookupMailExchange(self, domain):
                if domain == alias:
                    return defer.succeed(([RRHeader(name=domain, type=Record_CNAME.TYPE, payload=Record_CNAME(canonical))], [], []))
                elif domain == canonical:
                    return defer.succeed(([RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(0, exchange))], [], []))
                else:
                    return defer.fail(DNSNameError(domain))
        self.mx.resolver = DummyResolver()
        d = self.mx.getMX(alias)
        d.addCallback(self.assertEqual, Record_MX(name=exchange))
        return d

    def test_cnameChain(self):
        """
        If L{MXCalculator.getMX} encounters a CNAME chain which is longer than
        the length specified, the returned L{Deferred} should errback with
        L{CanonicalNameChainTooLong}.
        """

        class DummyResolver:
            """
            Fake resolver which generates a CNAME chain of infinite length in
            response to MX lookups.
            """
            chainCounter = 0

            def lookupMailExchange(self, domain):
                self.chainCounter += 1
                name = 'x-%d.example.com' % (self.chainCounter,)
                return defer.succeed(([RRHeader(name=domain, type=Record_CNAME.TYPE, payload=Record_CNAME(name))], [], []))
        cnameLimit = 3
        self.mx.resolver = DummyResolver()
        d = self.mx.getMX('mail.example.com', cnameLimit)
        self.assertFailure(d, twisted.mail.relaymanager.CanonicalNameChainTooLong)

        def cbChainTooLong(error):
            self.assertEqual(error.args[0], Record_CNAME('x-%d.example.com' % (cnameLimit + 1,)))
            self.assertEqual(self.mx.resolver.chainCounter, cnameLimit + 1)
        d.addCallback(cbChainTooLong)
        return d

    def test_cnameWithGlueRecords(self):
        """
        If an MX lookup returns a CNAME and the MX record for the CNAME, the
        L{Deferred} returned by L{MXCalculator.getMX} should be called back
        with the name from the MX record without further lookups being
        attempted.
        """
        lookedUp = []
        alias = 'alias.example.com'
        canonical = 'canonical.example.com'
        exchange = 'mail.example.com'

        class DummyResolver:

            def lookupMailExchange(self, domain):
                if domain != alias or lookedUp:
                    return ([], [], [])
                return defer.succeed(([RRHeader(name=alias, type=Record_CNAME.TYPE, payload=Record_CNAME(canonical)), RRHeader(name=canonical, type=Record_MX.TYPE, payload=Record_MX(name=exchange))], [], []))
        self.mx.resolver = DummyResolver()
        d = self.mx.getMX(alias)
        d.addCallback(self.assertEqual, Record_MX(name=exchange))
        return d

    def test_cnameLoopWithGlueRecords(self):
        """
        If an MX lookup returns two CNAME records which point to each other,
        the loop should be detected and the L{Deferred} returned by
        L{MXCalculator.getMX} should be errbacked with L{CanonicalNameLoop}.
        """
        firstAlias = 'cname1.example.com'
        secondAlias = 'cname2.example.com'

        class DummyResolver:

            def lookupMailExchange(self, domain):
                return defer.succeed(([RRHeader(name=firstAlias, type=Record_CNAME.TYPE, payload=Record_CNAME(secondAlias)), RRHeader(name=secondAlias, type=Record_CNAME.TYPE, payload=Record_CNAME(firstAlias))], [], []))
        self.mx.resolver = DummyResolver()
        d = self.mx.getMX(firstAlias)
        self.assertFailure(d, twisted.mail.relaymanager.CanonicalNameLoop)
        return d

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testManyRecords(self):
        self.auth.addresses['test.domain'] = ['mx1.test.domain', 'mx2.test.domain', 'mx3.test.domain']
        return self.mx.getMX('test.domain').addCallback(self._cbManyRecordsSuccessfulLookup)

    def _cbManyRecordsSuccessfulLookup(self, mx):
        self.assertTrue(str(mx.name).split('.', 1)[0] in ('mx1', 'mx2', 'mx3'))
        self.mx.markBad(str(mx.name))
        return self.mx.getMX('test.domain').addCallback(self._cbManyRecordsDifferentResult, mx)

    def _cbManyRecordsDifferentResult(self, nextMX, mx):
        self.assertNotEqual(str(mx.name), str(nextMX.name))
        self.mx.markBad(str(nextMX.name))
        return self.mx.getMX('test.domain').addCallback(self._cbManyRecordsLastResult, mx, nextMX)

    def _cbManyRecordsLastResult(self, lastMX, mx, nextMX):
        self.assertNotEqual(str(mx.name), str(lastMX.name))
        self.assertNotEqual(str(nextMX.name), str(lastMX.name))
        self.mx.markBad(str(lastMX.name))
        self.mx.markGood(str(nextMX.name))
        return self.mx.getMX('test.domain').addCallback(self._cbManyRecordsRepeatSpecificResult, nextMX)

    def _cbManyRecordsRepeatSpecificResult(self, againMX, nextMX):
        self.assertEqual(str(againMX.name), str(nextMX.name))