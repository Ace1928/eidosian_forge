import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_result_can_be_conflict(self):
    """A hook's result can be a conflict."""
    self.install_hook_conflict()
    builder = self.make_merge_builder()
    self.create_file_needing_contents_merge(builder, 'name1')
    conflicts = builder.merge(self.merge_type)
    self.assertEqual(1, len(conflicts))
    [conflict] = conflicts
    self.assertEqual('text conflict', conflict.typestring)
    if builder.this.supports_file_ids:
        self.assertEqual(conflict.file_id, builder.this.path2id('name1'))
    with builder.this.get_file('name1') as f:
        self.assertEqual(f.read(), b'text-with-conflict-markers-from-hook')