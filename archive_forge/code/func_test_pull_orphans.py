from breezy import tests
from breezy.revision import NULL_REVISION
from breezy.tests import per_workingtree
def test_pull_orphans(self):
    if not self.workingtree_format.missing_parent_conflicts:
        raise tests.TestSkipped('%r does not support missing parent conflicts' % self.workingtree_format)
    trunk = self.make_branch_deleting_dir('trunk')
    work = trunk.controldir.sprout('work', revision_id=b'2').open_workingtree()
    work.branch.get_config_stack().set('transform.orphan_policy', 'move')
    self.build_tree(['work/dir/foo', 'work/dir/subdir/', 'work/dir/subdir/foo'])
    work.pull(trunk)
    self.assertLength(0, work.conflicts())
    self.assertPathDoesNotExist('work/dir')