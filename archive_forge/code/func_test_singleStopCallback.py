from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_singleStopCallback(self):
    """
        L{task.react} doesn't try to stop the reactor if the L{defer.Deferred}
        the function it is passed is called back after the reactor has already
        been stopped.
        """

    async def main(reactor):
        reactor.callLater(1, reactor.stop)
        finished = defer.Deferred()
        reactor.addSystemEventTrigger('during', 'shutdown', finished.callback, None)
        return await finished
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
    self.assertEqual(r.seconds(), 1)
    self.assertEqual(0, exitError.code)