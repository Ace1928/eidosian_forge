import os
from breezy import tests
from breezy.tests import features
from breezy.transport import memory
def test_parallel_fork_unsupported(self):
    if getattr(os, 'fork', None) is not None:
        self.addCleanup(setattr, os, 'fork', os.fork)
        del os.fork
    out, err = self.run_bzr(['selftest', '--parallel=fork', '-s', 'bt.x'], retcode=3)
    self.assertIn('platform does not support fork', err)
    self.assertFalse(out)