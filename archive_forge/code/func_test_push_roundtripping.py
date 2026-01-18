import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_push_roundtripping(self):
    self.knownFailure('roundtripping is not yet supported')
    self.with_roundtripping()
    os.mkdir('bla')
    GitRepo.init(os.path.join(self.test_dir, 'bla'))
    self.run_bzr(['init', 'foo'])
    self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
    output, error = self.run_bzr(['push', '-d', 'foo', 'bla'])
    self.assertEqual(b'', output)
    self.assertTrue(error.endswith(b'Created new branch.\n'))