from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_pivot_root(self):
    """It's possible (albeit awkward) to move an existing dir to the root
        in a single snapshot by using unversion then flush then add.
        """
    builder = BranchBuilder(self.get_transport().clone('foo'))
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'orig-root', 'directory', None)), ('add', ('dir', b'dir-id', 'directory', None))], revision_id=b'A-id')
    builder.build_snapshot(None, [('unversion', ''), ('flush', None), ('add', ('', b'dir-id', 'directory', None))], revision_id=b'B-id')
    builder.finish_series()
    rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
    self.assertTreeShape([('', b'dir-id', 'directory')], rev_tree)