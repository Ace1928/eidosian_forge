import os
from breezy.errors import CommandError, NoSuchRevision
from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_revision_info_not_in_history(self):
    builder = self.make_branch_builder('branch')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
    builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
    builder.finish_series()
    self.check_output('  1 A-id\n??? B-id\n  2 C-id\n', 'revision-info -d branch revid:A-id revid:B-id revid:C-id')