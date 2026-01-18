from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_rename_out_of_unversioned_subdir(self):
    builder = self.build_a_rev()
    builder.build_snapshot(None, [('add', ('dir', b'dir-id', 'directory', None)), ('rename', ('a', 'dir/a'))], revision_id=b'B-id')
    builder.build_snapshot(None, [('rename', ('dir/a', 'a')), ('unversion', 'dir')], revision_id=b'C-id')
    rev_tree = builder.get_branch().repository.revision_tree(b'C-id')
    self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file')], rev_tree)