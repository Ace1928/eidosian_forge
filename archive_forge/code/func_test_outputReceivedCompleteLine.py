import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_outputReceivedCompleteLine(self):
    """
        Getting a complete output line on stdout generates a log message.
        """
    events = []
    self.addCleanup(globalLogPublisher.removeObserver, events.append)
    globalLogPublisher.addObserver(events.append)
    self.pm.addProcess('foo', ['foo'])
    self.pm.startService()
    self.reactor.advance(0)
    self.assertIn('foo', self.pm.protocols)
    self.reactor.advance(self.pm.threshold)
    self.pm.protocols['foo'].outReceived(b'hello world!\n')
    self.assertEquals(len(events), 1)
    namespace = events[0]['log_namespace']
    stream = events[0]['stream']
    tag = events[0]['tag']
    line = events[0]['line']
    self.assertEquals(namespace, 'twisted.runner.procmon.ProcessMonitor')
    self.assertEquals(stream, 'stdout')
    self.assertEquals(tag, 'foo')
    self.assertEquals(line, 'hello world!')