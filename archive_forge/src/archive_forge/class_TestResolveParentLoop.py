import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolveParentLoop(TestParametrizedResolveConflicts):
    _conflict_type = bzr_conflicts.ParentLoop
    _this_args = None
    _other_args = None
    scenarios = mirror_scenarios([(dict(_base_actions='create_dir1_dir2'), ('dir1_into_dir2', dict(actions='move_dir1_into_dir2', check='dir1_moved', dir_id=b'dir1-id', target_id=b'dir2-id', xfail=False)), ('dir2_into_dir1', dict(actions='move_dir2_into_dir1', check='dir2_moved', dir_id=b'dir2-id', target_id=b'dir1-id', xfail=False))), (dict(_base_actions='create_dir1_4'), ('dir1_into_dir4', dict(actions='move_dir1_into_dir4', check='dir1_2_moved', dir_id=b'dir1-id', target_id=b'dir4-id', xfail=True)), ('dir3_into_dir2', dict(actions='move_dir3_into_dir2', check='dir3_4_moved', dir_id=b'dir3-id', target_id=b'dir2-id', xfail=True)))])

    def do_create_dir1_dir2(self):
        return [('add', ('dir1', b'dir1-id', 'directory', '')), ('add', ('dir2', b'dir2-id', 'directory', ''))]

    def do_move_dir1_into_dir2(self):
        return [('rename', ('dir1', 'dir2/dir1'))]

    def check_dir1_moved(self):
        self.assertPathDoesNotExist('branch/dir1')
        self.assertPathExists('branch/dir2/dir1')

    def do_move_dir2_into_dir1(self):
        return [('rename', ('dir2', 'dir1/dir2'))]

    def check_dir2_moved(self):
        self.assertPathDoesNotExist('branch/dir2')
        self.assertPathExists('branch/dir1/dir2')

    def do_create_dir1_4(self):
        return [('add', ('dir1', b'dir1-id', 'directory', '')), ('add', ('dir1/dir2', b'dir2-id', 'directory', '')), ('add', ('dir3', b'dir3-id', 'directory', '')), ('add', ('dir3/dir4', b'dir4-id', 'directory', ''))]

    def do_move_dir1_into_dir4(self):
        return [('rename', ('dir1', 'dir3/dir4/dir1'))]

    def check_dir1_2_moved(self):
        self.assertPathDoesNotExist('branch/dir1')
        self.assertPathExists('branch/dir3/dir4/dir1')
        self.assertPathExists('branch/dir3/dir4/dir1/dir2')

    def do_move_dir3_into_dir2(self):
        return [('rename', ('dir3', 'dir1/dir2/dir3'))]

    def check_dir3_4_moved(self):
        self.assertPathDoesNotExist('branch/dir3')
        self.assertPathExists('branch/dir1/dir2/dir3')
        self.assertPathExists('branch/dir1/dir2/dir3/dir4')

    def _get_resolve_path_arg(self, wt, action):
        return wt.id2path(self._other['dir_id'])

    def assertParentLoop(self, wt, c):
        self.assertEqual(self._other['dir_id'], c.file_id)
        self.assertEqual(self._other['target_id'], c.conflict_file_id)
        if self._other['xfail']:
            self.knownFailure("ParentLoop doesn't carry enough info to resolve --take-other")
    _assert_conflict = assertParentLoop