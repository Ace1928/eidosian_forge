from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_runsUntilAsyncErrback(self):
    """
        L{task.react} runs the reactor until the L{defer.Deferred} returned by
        the function it is passed is errbacked, then it stops the reactor and
        reports the error.
        """

    class ExpectedException(Exception):
        pass

    async def main(reactor):
        finished = defer.Deferred()
        reactor.callLater(1, finished.errback, ExpectedException())
        return await finished
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
    self.assertEqual(1, exitError.code)
    errors = self.flushLoggedErrors(ExpectedException)
    self.assertEqual(len(errors), 1)