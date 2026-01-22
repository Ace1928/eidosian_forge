import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolveDuplicateEntry(TestParametrizedResolveConflicts):
    _conflict_type = bzr_conflicts.DuplicateEntry
    scenarios = mirror_scenarios([(dict(_base_actions='nothing'), ('filea_created', dict(actions='create_file_a', check='file_content_a', path='file', file_id=b'file-a-id')), ('fileb_created', dict(actions='create_file_b', check='file_content_b', path='file', file_id=b'file-b-id'))), (dict(_base_actions='create_file_a'), ('filea_replaced', dict(actions='replace_file_a_by_b', check='file_content_b', path='file', file_id=b'file-b-id')), ('filea_modified', dict(actions='modify_file_a', check='file_new_content', path='file', file_id=b'file-a-id')))])

    def do_nothing(self):
        return []

    def do_create_file_a(self):
        return [('add', ('file', b'file-a-id', 'file', b'file a content\n'))]

    def check_file_content_a(self):
        self.assertFileEqual(b'file a content\n', 'branch/file')

    def do_create_file_b(self):
        return [('add', ('file', b'file-b-id', 'file', b'file b content\n'))]

    def check_file_content_b(self):
        self.assertFileEqual(b'file b content\n', 'branch/file')

    def do_replace_file_a_by_b(self):
        return [('unversion', 'file'), ('add', ('file', b'file-b-id', 'file', b'file b content\n'))]

    def do_modify_file_a(self):
        return [('modify', ('file', b'new content\n'))]

    def check_file_new_content(self):
        self.assertFileEqual(b'new content\n', 'branch/file')

    def _get_resolve_path_arg(self, wt, action):
        return self._this['path']

    def assertDuplicateEntry(self, wt, c):
        tpath = self._this['path']
        tfile_id = self._this['file_id']
        opath = self._other['path']
        ofile_id = self._other['file_id']
        self.assertEqual(tpath, opath)
        self.assertEqual(tfile_id, c.file_id)
        self.assertEqual(tpath + '.moved', c.path)
        self.assertEqual(tpath, c.conflict_path)
    _assert_conflict = assertDuplicateEntry