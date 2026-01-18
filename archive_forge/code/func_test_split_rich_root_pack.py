from breezy import tests, workingtree
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack4
from breezy.bzr.knitrepo import RepositoryFormatKnit4
def test_split_rich_root_pack(self):
    self.split_formats('rich-root-pack', RepositoryFormatKnitPack4)