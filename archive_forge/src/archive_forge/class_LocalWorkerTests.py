import os
from io import BytesIO, StringIO
from typing import Type
from unittest import TestCase as PyUnitTestCase
from zope.interface.verify import verifyObject
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.internet.defer import Deferred, fail
from twisted.internet.error import ConnectionLost, ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.trial._dist import managercommands
from twisted.trial._dist.worker import (
from twisted.trial.reporter import TestResult
from twisted.trial.test import pyunitcases, skipping
from twisted.trial.unittest import TestCase, makeTodo
from .matchers import isFailure, matches_result, similarFrame
class LocalWorkerTests(TestCase):
    """
    Tests for L{LocalWorker} and L{LocalWorkerTransport}.
    """

    def tidyLocalWorker(self, *args, **kwargs):
        """
        Create a L{LocalWorker}, connect it to a transport, and ensure
        its log files are closed.

        @param args: See L{LocalWorker}

        @param kwargs: See L{LocalWorker}

        @return: a L{LocalWorker} instance
        """
        worker = LocalWorker(*args, **kwargs)
        worker.makeConnection(FakeTransport())
        self.addCleanup(worker._outLog.close)
        self.addCleanup(worker._errLog.close)
        return worker

    def test_exitBeforeConnected(self):
        """
        L{LocalWorker.exit} fails with L{NotRunning} if it is called before the
        protocol is connected to a transport.
        """
        worker = LocalWorker(SpyDataLocalWorkerAMP(), FilePath(self.mktemp()), StringIO())
        self.failureResultOf(worker.exit(), NotRunning)

    def test_exitAfterDisconnected(self):
        """
        L{LocalWorker.exit} fails with L{NotRunning} if it is called after the the
        protocol is disconnected from its transport.
        """
        worker = self.tidyLocalWorker(SpyDataLocalWorkerAMP(), FilePath(self.mktemp()), StringIO())
        worker.processEnded(Failure(ProcessDone(0)))
        self.failureResultOf(worker.endDeferred, ProcessDone)
        self.failureResultOf(worker.exit(), NotRunning)

    def test_childDataReceived(self):
        """
        L{LocalWorker.childDataReceived} forwards the received data to linked
        L{AMP} protocol if the right file descriptor, otherwise forwards to
        C{ProcessProtocol.childDataReceived}.
        """
        localWorker = self.tidyLocalWorker(SpyDataLocalWorkerAMP(), FilePath(self.mktemp()), 'test.log')
        localWorker._outLog = BytesIO()
        localWorker.childDataReceived(4, b'foo')
        localWorker.childDataReceived(1, b'bar')
        self.assertEqual(b'foo', localWorker._ampProtocol.dataString)
        self.assertEqual(b'bar', localWorker._outLog.getvalue())

    def test_newlineStyle(self):
        """
        L{LocalWorker} writes the log data with local newlines.
        """
        amp = SpyDataLocalWorkerAMP()
        tempDir = FilePath(self.mktemp())
        tempDir.makedirs()
        logPath = tempDir.child('test.log')
        with open(logPath.path, 'wt', encoding='utf-8') as logFile:
            worker = LocalWorker(amp, tempDir, logFile)
            worker.makeConnection(FakeTransport())
            self.addCleanup(worker._outLog.close)
            self.addCleanup(worker._errLog.close)
            expected = 'Here comes the ☉!'
            amp.testWrite(expected)
        self.assertEqual(expected + os.linesep, logPath.getContent().decode('utf-8'))

    def test_outReceived(self):
        """
        L{LocalWorker.outReceived} logs the output into its C{_outLog} log
        file.
        """
        localWorker = self.tidyLocalWorker(SpyDataLocalWorkerAMP(), FilePath(self.mktemp()), 'test.log')
        localWorker._outLog = BytesIO()
        data = b'The quick brown fox jumps over the lazy dog'
        localWorker.outReceived(data)
        self.assertEqual(data, localWorker._outLog.getvalue())

    def test_errReceived(self):
        """
        L{LocalWorker.errReceived} logs the errors into its C{_errLog} log
        file.
        """
        localWorker = self.tidyLocalWorker(SpyDataLocalWorkerAMP(), FilePath(self.mktemp()), 'test.log')
        localWorker._errLog = BytesIO()
        data = b'The quick brown fox jumps over the lazy dog'
        localWorker.errReceived(data)
        self.assertEqual(data, localWorker._errLog.getvalue())

    def test_write(self):
        """
        L{LocalWorkerTransport.write} forwards the written data to the given
        transport.
        """
        transport = FakeTransport()
        localTransport = LocalWorkerTransport(transport)
        data = b'The quick brown fox jumps over the lazy dog'
        localTransport.write(data)
        self.assertEqual(data, transport.dataString)

    def test_writeSequence(self):
        """
        L{LocalWorkerTransport.writeSequence} forwards the written data to the
        given transport.
        """
        transport = FakeTransport()
        localTransport = LocalWorkerTransport(transport)
        data = (b'The quick ', b'brown fox jumps ', b'over the lazy dog')
        localTransport.writeSequence(data)
        self.assertEqual(b''.join(data), transport.dataString)

    def test_loseConnection(self):
        """
        L{LocalWorkerTransport.loseConnection} forwards the call to the given
        transport.
        """
        transport = FakeTransport()
        localTransport = LocalWorkerTransport(transport)
        localTransport.loseConnection()
        self.assertEqual(transport.calls, 1)

    def test_connectionLost(self):
        """
        L{LocalWorker.connectionLost} closes the per-worker log streams.
        """
        localWorker = self.tidyLocalWorker(SpyDataLocalWorkerAMP(), FilePath(self.mktemp()), 'test.log')
        localWorker.connectionLost(None)
        self.assertTrue(localWorker._outLog.closed)
        self.assertTrue(localWorker._errLog.closed)

    def test_processEnded(self):
        """
        L{LocalWorker.processEnded} calls C{connectionLost} on itself and on
        the L{AMP} protocol.
        """
        transport = FakeTransport()
        protocol = SpyDataLocalWorkerAMP()
        localWorker = LocalWorker(protocol, FilePath(self.mktemp()), 'test.log')
        localWorker.makeConnection(transport)
        localWorker.processEnded(Failure(ProcessDone(0)))
        self.assertTrue(localWorker._outLog.closed)
        self.assertTrue(localWorker._errLog.closed)
        self.assertIdentical(None, protocol.transport)
        return self.assertFailure(localWorker.endDeferred, ProcessDone)

    def test_addresses(self):
        """
        L{LocalWorkerTransport.getPeer} and L{LocalWorkerTransport.getHost}
        return L{IAddress} objects.
        """
        localTransport = LocalWorkerTransport(None)
        self.assertTrue(verifyObject(IAddress, localTransport.getPeer()))
        self.assertTrue(verifyObject(IAddress, localTransport.getHost()))

    def test_transport(self):
        """
        L{LocalWorkerTransport} implements L{ITransport} to be able to be used
        by L{AMP}.
        """
        localTransport = LocalWorkerTransport(None)
        self.assertTrue(verifyObject(ITransport, localTransport))

    def test_startError(self):
        """
        L{LocalWorker} swallows the exceptions returned by the L{AMP} protocol
        start method, as it generates unnecessary errors.
        """

        def failCallRemote(command, directory):
            return fail(RuntimeError('oops'))
        protocol = SpyDataLocalWorkerAMP()
        protocol.callRemote = failCallRemote
        self.tidyLocalWorker(protocol, FilePath(self.mktemp()), 'test.log')
        self.assertEqual([], self.flushLoggedErrors(RuntimeError))