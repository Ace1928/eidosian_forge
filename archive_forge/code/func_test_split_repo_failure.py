from breezy import tests, workingtree
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack4
from breezy.bzr.knitrepo import RepositoryFormatKnit4
def test_split_repo_failure(self):
    repo = self.make_repository('branch', shared=True, format='knit')
    a_branch = repo.controldir.create_branch()
    self.build_tree(['a/', 'a/b/', 'a/b/c/', 'a/b/c/d'])
    wt = a_branch.create_checkout('a', lightweight=True)
    wt.add(['b', 'b/c', 'b/c/d'], ids=[b'b-id', b'c-id', b'd-id'])
    wt.commit('added files')
    self.run_bzr_error(('must upgrade your branch at .*a',), 'split a/b')