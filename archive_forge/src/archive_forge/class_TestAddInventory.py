from breezy import errors
from breezy.repository import WriteGroup
from breezy.tests.per_repository_reference import \
class TestAddInventory(TestCaseWithExternalReferenceRepository):

    def test_add_inventory_goes_to_repo(self):
        tree = self.make_branch_and_tree('sample')
        revid = tree.commit('one')
        inv = tree.branch.repository.get_inventory(revid)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        base = self.make_repository('base')
        repo = self.make_referring('referring', base)
        with repo.lock_write(), WriteGroup(repo):
            repo.add_inventory(revid, inv, [])
        repo.lock_read()
        self.addCleanup(repo.unlock)
        inv2 = repo.get_inventory(revid)
        content1 = {file_id: inv.get_entry(file_id) for file_id in inv.iter_all_ids()}
        content2 = {file_id: inv.get_entry(file_id) for file_id in inv2.iter_all_ids()}
        self.assertEqual(content1, content2)
        self.assertRaises(errors.NoSuchRevision, base.get_inventory, revid)