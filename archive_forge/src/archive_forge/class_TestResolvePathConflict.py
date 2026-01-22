import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolvePathConflict(TestParametrizedResolveConflicts):
    _conflict_type = bzr_conflicts.PathConflict

    def do_nothing(self):
        return []
    scenarios = mirror_scenarios([(dict(_base_actions='create_file'), ('file_renamed', dict(actions='rename_file', check='file_renamed', path='new-file', file_id=b'file-id')), ('file_deleted', dict(actions='delete_file', check='file_doesnt_exist', path='<deleted>', file_id=b'file-id'))), (dict(_base_actions='create_file_in_dir'), ('file_renamed_in_dir', dict(actions='rename_file_in_dir', check='file_in_dir_renamed', path='dir/new-file', file_id=b'file-id')), ('file_deleted', dict(actions='delete_file_in_dir', check='file_in_dir_doesnt_exist', path='<deleted>', file_id=b'file-id'))), (dict(_base_actions='create_file'), ('file_renamed', dict(actions='rename_file', check='file_renamed', path='new-file', file_id=b'file-id')), ('file_renamed2', dict(actions='rename_file2', check='file_renamed2', path='new-file2', file_id=b'file-id'))), (dict(_base_actions='create_dir'), ('dir_renamed', dict(actions='rename_dir', check='dir_renamed', path='new-dir', file_id=b'dir-id')), ('dir_deleted', dict(actions='delete_dir', check='dir_doesnt_exist', path='<deleted>', file_id=b'dir-id'))), (dict(_base_actions='create_dir'), ('dir_renamed', dict(actions='rename_dir', check='dir_renamed', path='new-dir', file_id=b'dir-id')), ('dir_renamed2', dict(actions='rename_dir2', check='dir_renamed2', path='new-dir2', file_id=b'dir-id')))])

    def do_create_file(self):
        return [('add', ('file', b'file-id', 'file', b'trunk content\n'))]

    def do_create_dir(self):
        return [('add', ('dir', b'dir-id', 'directory', ''))]

    def do_rename_file(self):
        return [('rename', ('file', 'new-file'))]

    def check_file_renamed(self):
        self.assertPathDoesNotExist('branch/file')
        self.assertPathExists('branch/new-file')

    def do_rename_file2(self):
        return [('rename', ('file', 'new-file2'))]

    def check_file_renamed2(self):
        self.assertPathDoesNotExist('branch/file')
        self.assertPathExists('branch/new-file2')

    def do_rename_dir(self):
        return [('rename', ('dir', 'new-dir'))]

    def check_dir_renamed(self):
        self.assertPathDoesNotExist('branch/dir')
        self.assertPathExists('branch/new-dir')

    def do_rename_dir2(self):
        return [('rename', ('dir', 'new-dir2'))]

    def check_dir_renamed2(self):
        self.assertPathDoesNotExist('branch/dir')
        self.assertPathExists('branch/new-dir2')

    def do_delete_file(self):
        return [('unversion', 'file')]

    def do_delete_file_in_dir(self):
        return [('unversion', 'dir/file')]

    def check_file_doesnt_exist(self):
        self.assertPathDoesNotExist('branch/file')

    def do_delete_dir(self):
        return [('unversion', 'dir')]

    def check_dir_doesnt_exist(self):
        self.assertPathDoesNotExist('branch/dir')

    def do_create_file_in_dir(self):
        return [('add', ('dir', b'dir-id', 'directory', '')), ('add', ('dir/file', b'file-id', 'file', b'trunk content\n'))]

    def do_rename_file_in_dir(self):
        return [('rename', ('dir/file', 'dir/new-file'))]

    def check_file_in_dir_renamed(self):
        self.assertPathDoesNotExist('branch/dir/file')
        self.assertPathExists('branch/dir/new-file')

    def check_file_in_dir_doesnt_exist(self):
        self.assertPathDoesNotExist('branch/dir/file')

    def _get_resolve_path_arg(self, wt, action):
        tpath = self._this['path']
        opath = self._other['path']
        if tpath == '<deleted>':
            path = opath
        else:
            path = tpath
        return path

    def assertPathConflict(self, wt, c):
        tpath = self._this['path']
        tfile_id = self._this['file_id']
        opath = self._other['path']
        ofile_id = self._other['file_id']
        self.assertEqual(tfile_id, ofile_id)
        self.assertEqual(tfile_id, c.file_id)
        self.assertEqual(tpath, c.path)
        self.assertEqual(opath, c.conflict_path)
    _assert_conflict = assertPathConflict