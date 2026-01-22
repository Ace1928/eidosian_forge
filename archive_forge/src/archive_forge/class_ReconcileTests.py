import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
class ReconcileTests(ExternalBase):

    def test_simple_reconcile(self):
        tree = self.make_branch_and_tree('.', format='git')
        self.build_tree_contents([('a', 'text for a\n')])
        tree.add(['a'])
        output, error = self.run_bzr('reconcile')
        self.assertContainsRe(output, 'Reconciling branch file://.*\nReconciling repository file://.*\nReconciliation complete.\n')
        self.assertEqual(error, '')