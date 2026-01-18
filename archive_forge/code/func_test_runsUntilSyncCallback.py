from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_runsUntilSyncCallback(self):
    """
        L{task.react} returns quickly if the L{Deferred} returned by the
        function it is passed has already been called back at the time it is
        returned.
        """

    async def main(reactor):
        return await defer.succeed(None)
    r = _FakeReactor()
    exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
    self.assertEqual(0, exitError.code)
    self.assertEqual(r.seconds(), 0)