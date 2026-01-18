from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_two_fetches(self):
    self.make_two_commits(change_root=False, fetch_twice=True)
    self.assertEqual(((b'TREE_ROOT', b'first-id'),), self.get_parents(b'TREE_ROOT', b'second-id'))