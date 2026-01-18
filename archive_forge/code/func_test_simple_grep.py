import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
def test_simple_grep(self):
    tree = self.make_branch_and_tree('.', format='git')
    self.build_tree_contents([('a', 'text for a\n')])
    tree.add(['a'])
    output, error = self.run_bzr('grep text')
    self.assertEqual(output, 'a:text for a\n')
    self.assertEqual(error, '')