import os
from breezy import tests
from breezy.bzr.tests.matchers import ContainsNoVfsCalls
from breezy.errors import NoSuchRevision
def test_stale_revno_tree(self):
    builder = self.make_branch_builder('branch')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
    builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
    builder.finish_series()
    b = builder.get_branch()
    co_b = b.create_checkout('checkout_b', lightweight=True, revision_id=b'B-id')
    out, err = self.run_bzr('revno checkout_b')
    self.assertEqual('', err)
    self.assertEqual('2\n', out)
    out, err = self.run_bzr('revno --tree checkout_b')
    self.assertEqual('', err)
    self.assertEqual('???\n', out)