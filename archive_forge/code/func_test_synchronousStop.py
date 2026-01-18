from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_synchronousStop(self):
    """
        L{task.react} handles when the reactor is stopped just before the
        returned L{Deferred} fires.
        """

    async def main(reactor):
        d = defer.Deferred()

        def stop():
            reactor.stop()
            d.callback(None)
        reactor.callWhenRunning(stop)
        return await d
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, [], _reactor=r)
    self.assertEqual(0, exitError.code)