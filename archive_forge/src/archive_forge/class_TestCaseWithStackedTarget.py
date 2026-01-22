from breezy import errors, tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
class TestCaseWithStackedTarget(TestCaseWithRepository):
    r1_key = (b'rev1-id',)
    r2_key = (b'rev2-id',)

    def make_stacked_target(self):
        base_tree = self.make_branch_and_tree('base')
        self.build_tree(['base/f1.txt'])
        base_tree.add(['f1.txt'], ids=[b'f1.txt-id'])
        base_tree.commit('initial', rev_id=self.r1_key[0])
        self.build_tree(['base/f2.txt'])
        base_tree.add(['f2.txt'], ids=[b'f2.txt-id'])
        base_tree.commit('base adds f2', rev_id=self.r2_key[0])
        stacked_url = urlutils.join(base_tree.branch.base, '../stacked')
        stacked_bzrdir = base_tree.controldir.sprout(stacked_url, stacked=True)
        if isinstance(stacked_bzrdir, remote.RemoteBzrDir):
            stacked_branch = stacked_bzrdir.open_branch()
            stacked_tree = stacked_branch.create_checkout('stacked', lightweight=True)
        else:
            stacked_tree = stacked_bzrdir.open_workingtree()
        return (base_tree, stacked_tree)