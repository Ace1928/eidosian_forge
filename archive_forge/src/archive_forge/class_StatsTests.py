import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
class StatsTests(ExternalBase):

    def test_simple_stats(self):
        self.requireFeature(PluginLoadedFeature('stats'))
        tree = self.make_branch_and_tree('.', format='git')
        self.build_tree_contents([('a', 'text for a\n')])
        tree.add(['a'])
        tree.commit('a commit', committer='Somebody <somebody@example.com>')
        output, error = self.run_bzr('stats')
        self.assertEqual(output, '   1 Somebody <somebody@example.com>\n')