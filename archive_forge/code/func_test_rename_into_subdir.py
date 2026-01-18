from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_rename_into_subdir(self):
    builder = self.build_a_rev()
    builder.build_snapshot(None, [('add', ('dir', b'dir-id', 'directory', None)), ('rename', ('a', 'dir/a'))], revision_id=b'B-id')
    rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
    self.assertTreeShape([('', b'a-root-id', 'directory'), ('dir', b'dir-id', 'directory'), ('dir/a', b'a-id', 'file')], rev_tree)