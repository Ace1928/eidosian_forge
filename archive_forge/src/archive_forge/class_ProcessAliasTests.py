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
class ProcessAliasTests(TestCase):
    """
    Tests for alias resolution.
    """
    if interfaces.IReactorProcess(reactor, None) is None:
        skip = 'IReactorProcess not supported'
    lines = ['First line', 'Next line', '', 'After a blank line', 'Last line']

    def exitStatus(self, code):
        """
        Construct a status from the given exit code.

        @type code: L{int} between 0 and 255 inclusive.
        @param code: The exit status which the code will represent.

        @rtype: L{int}
        @return: A status integer for the given exit code.
        """
        status = code << 8 | 0
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(os.WEXITSTATUS(status), code)
        self.assertFalse(os.WIFSIGNALED(status))
        return status

    def signalStatus(self, signal):
        """
        Construct a status from the given signal.

        @type signal: L{int} between 0 and 255 inclusive.
        @param signal: The signal number which the status will represent.

        @rtype: L{int}
        @return: A status integer for the given signal.
        """
        status = signal
        self.assertTrue(os.WIFSIGNALED(status))
        self.assertEqual(os.WTERMSIG(status), signal)
        self.assertFalse(os.WIFEXITED(status))
        return status

    def setUp(self):
        """
        Replace L{smtp.DNSNAME} with a well-known value.
        """
        self.DNSNAME = smtp.DNSNAME
        smtp.DNSNAME = ''

    def tearDown(self):
        """
        Restore the original value of L{smtp.DNSNAME}.
        """
        smtp.DNSNAME = self.DNSNAME

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_processAlias(self):
        """
        Standard call to C{mail.alias.ProcessAlias}: check that the specified
        script is called, and that the input is correctly transferred to it.
        """
        sh = FilePath(self.mktemp())
        sh.setContent('#!/bin/sh\nrm -f process.alias.out\nwhile read i; do\n    echo $i >> process.alias.out\ndone')
        os.chmod(sh.path, 448)
        a = mail.alias.ProcessAlias(sh.path, None, None)
        m = a.createMessageReceiver()
        for l in self.lines:
            m.lineReceived(l)

        def _cbProcessAlias(ignored):
            with open('process.alias.out') as f:
                lines = f.readlines()
            self.assertEqual([L[:-1] for L in lines], self.lines)
        return m.eomReceived().addCallback(_cbProcessAlias)

    def test_processAliasTimeout(self):
        """
        If the alias child process does not exit within a particular period of
        time, the L{Deferred} returned by L{MessageWrapper.eomReceived} should
        fail with L{ProcessAliasTimeout} and send the I{KILL} signal to the
        child process..
        """
        reactor = task.Clock()
        transport = StubProcess()
        proto = mail.alias.ProcessAliasProtocol()
        proto.makeConnection(transport)
        receiver = mail.alias.MessageWrapper(proto, None, reactor)
        d = receiver.eomReceived()
        reactor.advance(receiver.completionTimeout)

        def timedOut(ignored):
            self.assertEqual(transport.signals, ['KILL'])
            proto.processEnded(ProcessTerminated(self.signalStatus(signal.SIGKILL)))
        self.assertFailure(d, mail.alias.ProcessAliasTimeout)
        d.addCallback(timedOut)
        return d

    def test_earlyProcessTermination(self):
        """
        If the process associated with an L{mail.alias.MessageWrapper} exits
        before I{eomReceived} is called, the L{Deferred} returned by
        I{eomReceived} should fail.
        """
        transport = StubProcess()
        protocol = mail.alias.ProcessAliasProtocol()
        protocol.makeConnection(transport)
        receiver = mail.alias.MessageWrapper(protocol, None, None)
        protocol.processEnded(failure.Failure(ProcessDone(0)))
        return self.assertFailure(receiver.eomReceived(), ProcessDone)

    def _terminationTest(self, status):
        """
        Verify that if the process associated with an
        L{mail.alias.MessageWrapper} exits with the given status, the
        L{Deferred} returned by I{eomReceived} fails with L{ProcessTerminated}.
        """
        transport = StubProcess()
        protocol = mail.alias.ProcessAliasProtocol()
        protocol.makeConnection(transport)
        receiver = mail.alias.MessageWrapper(protocol, None, None)
        protocol.processEnded(failure.Failure(ProcessTerminated(status)))
        return self.assertFailure(receiver.eomReceived(), ProcessTerminated)

    def test_errorProcessTermination(self):
        """
        If the process associated with an L{mail.alias.MessageWrapper} exits
        with a non-zero exit code, the L{Deferred} returned by I{eomReceived}
        should fail.
        """
        return self._terminationTest(self.exitStatus(1))

    def test_signalProcessTermination(self):
        """
        If the process associated with an L{mail.alias.MessageWrapper} exits
        because it received a signal, the L{Deferred} returned by
        I{eomReceived} should fail.
        """
        return self._terminationTest(self.signalStatus(signal.SIGHUP))

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_aliasResolution(self):
        """
        Check that the C{resolve} method of alias processors produce the correct
        set of objects:
            - direct alias with L{mail.alias.AddressAlias} if a simple input is passed
            - aliases in a file with L{mail.alias.FileWrapper} if an input in the format
              '/file' is given
            - aliases resulting of a process call wrapped by L{mail.alias.MessageWrapper}
              if the format is '|process'
        """
        aliases = {}
        domain = {'': TestDomain(aliases, ['user1', 'user2', 'user3'])}
        A1 = MockAliasGroup(['user1', '|echo', '/file'], domain, 'alias1')
        A2 = MockAliasGroup(['user2', 'user3'], domain, 'alias2')
        A3 = mail.alias.AddressAlias('alias1', domain, 'alias3')
        aliases.update({'alias1': A1, 'alias2': A2, 'alias3': A3})
        res1 = A1.resolve(aliases)
        r1 = map(str, res1.objs)
        r1.sort()
        expected = map(str, [mail.alias.AddressAlias('user1', None, None), mail.alias.MessageWrapper(DummyProcess(), 'echo'), mail.alias.FileWrapper('/file')])
        expected.sort()
        self.assertEqual(r1, expected)
        res2 = A2.resolve(aliases)
        r2 = map(str, res2.objs)
        r2.sort()
        expected = map(str, [mail.alias.AddressAlias('user2', None, None), mail.alias.AddressAlias('user3', None, None)])
        expected.sort()
        self.assertEqual(r2, expected)
        res3 = A3.resolve(aliases)
        r3 = map(str, res3.objs)
        r3.sort()
        expected = map(str, [mail.alias.AddressAlias('user1', None, None), mail.alias.MessageWrapper(DummyProcess(), 'echo'), mail.alias.FileWrapper('/file')])
        expected.sort()
        self.assertEqual(r3, expected)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_cyclicAlias(self):
        """
        Check that a cycle in alias resolution is correctly handled.
        """
        aliases = {}
        domain = {'': TestDomain(aliases, [])}
        A1 = mail.alias.AddressAlias('alias2', domain, 'alias1')
        A2 = mail.alias.AddressAlias('alias3', domain, 'alias2')
        A3 = mail.alias.AddressAlias('alias1', domain, 'alias3')
        aliases.update({'alias1': A1, 'alias2': A2, 'alias3': A3})
        self.assertEqual(aliases['alias1'].resolve(aliases), None)
        self.assertEqual(aliases['alias2'].resolve(aliases), None)
        self.assertEqual(aliases['alias3'].resolve(aliases), None)
        A4 = MockAliasGroup(['|echo', 'alias1'], domain, 'alias4')
        aliases['alias4'] = A4
        res = A4.resolve(aliases)
        r = map(str, res.objs)
        r.sort()
        expected = map(str, [mail.alias.MessageWrapper(DummyProcess(), 'echo')])
        expected.sort()
        self.assertEqual(r, expected)