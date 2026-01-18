import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_push_lossy_non_mainline_incremental(self):
    self.run_bzr(['init', '--git', 'bla'])
    self.run_bzr(['init', 'foo'])
    self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
    self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
    output, error = self.run_bzr(['push', '--lossy', '-d', 'foo', 'bla'])
    self.assertEqual('', output)
    self.assertEqual('Pushing from a Bazaar to a Git repository. For better performance, push into a Bazaar repository.\nAll changes applied successfully.\nPushed up to revision 2.\n', error)
    self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
    output, error = self.run_bzr(['push', '--lossy', '-d', 'foo', 'bla'])
    self.assertEqual('', output)
    self.assertEqual('Pushing from a Bazaar to a Git repository. For better performance, push into a Bazaar repository.\nAll changes applied successfully.\nPushed up to revision 3.\n', error)