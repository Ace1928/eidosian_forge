import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
@onlyOnPOSIX
def test_process_unregistered_before_protocol_ended_callback(self):
    """
        Process is removed from reapProcessHandler dict before running
        ProcessProtocol.processEnded() callback.
        """
    results = []

    class TestProcessProtocol(ProcessProtocol):
        """
            Process protocol captures own presence in
            process.reapProcessHandlers at time of .processEnded() callback.

            @ivar deferred: A deferred fired when the .processEnded() callback
                has completed.
            @type deferred: L{Deferred<defer.Deferred>}
            """

        def __init__(self):
            self.deferred = Deferred()

        def processEnded(self, status):
            """
                Capture whether the process has already been removed
                from process.reapProcessHandlers.

                @param status: unused
                """
            from twisted.internet import process
            handlers = process.reapProcessHandlers
            processes = handlers.values()
            if self.transport in processes:
                results.append('process present but should not be')
            else:
                results.append('process already removed as desired')
            self.deferred.callback(None)

    @inlineCallbacks
    def launchProcessAndWait(reactor):
        """
            Launch and wait for a subprocess and allow the TestProcessProtocol
            to capture the order of the .processEnded() callback vs. removal
            from process.reapProcessHandlers.

            @param reactor: Reactor used to spawn the test process and to be
                stopped when checks are complete.
            @type reactor: object providing
                L{twisted.internet.interfaces.IReactorProcess} and
                L{twisted.internet.interfaces.IReactorCore}.
            """
        try:
            testProcessProtocol = TestProcessProtocol()
            reactor.spawnProcess(testProcessProtocol, pyExe, [pyExe, '--version'])
            yield testProcessProtocol.deferred
        except Exception as e:
            results.append(e)
        finally:
            reactor.stop()
    reactor = self.buildReactor()
    reactor.callWhenRunning(launchProcessAndWait, reactor)
    self.runReactor(reactor)
    hamcrest.assert_that(results, hamcrest.equal_to(['process already removed as desired']))