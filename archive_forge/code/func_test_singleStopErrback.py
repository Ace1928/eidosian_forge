from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_singleStopErrback(self):
    """
        L{task.react} doesn't try to stop the reactor if the L{defer.Deferred}
        the function it is passed is errbacked after the reactor has already
        been stopped.
        """

    class ExpectedException(Exception):
        pass

    async def main(reactor):
        reactor.callLater(1, reactor.stop)
        finished = defer.Deferred()
        reactor.addSystemEventTrigger('during', 'shutdown', finished.errback, ExpectedException())
        return await finished
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
    self.assertEqual(1, exitError.code)
    self.assertEqual(r.seconds(), 1)
    errors = self.flushLoggedErrors(ExpectedException)
    self.assertEqual(len(errors), 1)