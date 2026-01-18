from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_unknown_action(self):
    builder = self.build_a_rev()
    e = self.assertRaises(ValueError, builder.build_snapshot, None, [('weirdo', ('foo',))], revision_id=b'B-id')
    self.assertEqual('Unknown build action: "weirdo"', str(e))