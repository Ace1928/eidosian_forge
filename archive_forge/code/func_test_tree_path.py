import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def test_tree_path(self):
    tree = self.make_branch_and_tree('.', format=self._format)
    tree.mkdir('subdir')
    tree.commit('Add subdir')
    with Workspace(tree, use_inotify=self._use_inotify) as ws:
        self.assertEqual('foo', ws.tree_path('foo'))
        self.assertEqual('', ws.tree_path())
    with Workspace(tree, subpath='subdir', use_inotify=self._use_inotify) as ws:
        self.assertEqual('subdir/foo', ws.tree_path('foo'))
        self.assertEqual('subdir/', ws.tree_path())