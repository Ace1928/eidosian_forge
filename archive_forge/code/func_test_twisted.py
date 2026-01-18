import sys
from io import StringIO
from twisted.internet import defer, reactor
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import TestCase
def test_twisted(self):
    """Invoking python -m twisted should execute twist."""
    cmd = sys.executable
    p = Accumulator()
    d = p.endedDeferred = defer.Deferred()
    reactor.spawnProcess(p, cmd, [cmd, '-m', 'twisted', '--help'], env=None)
    p.transport.closeStdin()
    from twisted import __main__
    self.patch(sys, 'argv', [__main__.__file__, '--help'])

    def processEnded(ign):
        f = p.outF
        output = f.getvalue()
        self.assertTrue(b'-m twisted [options] plugin [plugin_options]' in output, output)
    return d.addCallback(processEnded)