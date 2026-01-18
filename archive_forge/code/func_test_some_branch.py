from ... import tests
from ..object_store import BazaarObjectStore
from ..refs import BazaarRefsContainer, branch_name_to_ref, ref_to_branch_name
def test_some_branch(self):
    tree = self.make_branch_and_tree('.')
    revid = tree.commit('somechange')
    otherbranch = tree.controldir.create_branch(name='otherbranch')
    otherbranch.generate_revision_history(revid)
    store = BazaarObjectStore(tree.branch.repository)
    refs = BazaarRefsContainer(tree.controldir, store)
    self.assertEqual(refs.as_dict(), {b'HEAD': store._lookup_revision_sha1(revid), b'refs/heads/otherbranch': store._lookup_revision_sha1(revid)})