import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def test_root_add(self):
    tree = self.make_branch_and_tree('.', format=self._format)
    with Workspace(tree, use_inotify=self._use_inotify) as ws:
        self.build_tree_contents([('afile', 'somecontents')])
        changes = [c for c in ws.iter_changes() if c.path[1] != '']
        self.assertEqual(1, len(changes), changes)
        self.assertEqual((None, 'afile'), changes[0].path)
        ws.commit(message='Commit message')
        self.assertEqual(list(ws.iter_changes()), [])
        self.build_tree_contents([('afile', 'newcontents')])
        [change] = list(ws.iter_changes())
        self.assertEqual(('afile', 'afile'), change.path)