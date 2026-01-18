import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_push_lossy_non_mainline(self):
    self.run_bzr(['init', '--git', 'bla'])
    self.run_bzr(['init', 'foo'])
    self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
    self.run_bzr(['branch', 'foo', 'foo1'])
    self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo1'])
    self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
    self.run_bzr(['merge', '-d', 'foo', 'foo1'])
    self.run_bzr(['commit', '--unchanged', '-m', 'merge', 'foo'])
    output, error = self.run_bzr(['push', '--lossy', '-r1.1.1', '-d', 'foo', 'bla'])
    self.assertEqual('', output)
    self.assertEqual('Pushing from a Bazaar to a Git repository. For better performance, push into a Bazaar repository.\nAll changes applied successfully.\nPushed up to revision 2.\n', error)