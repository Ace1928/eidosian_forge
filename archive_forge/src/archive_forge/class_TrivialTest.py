from breezy import controldir, tests
from breezy.bzr import inventory
from breezy.repository import WriteGroup
class TrivialTest(tests.TestCaseWithTransport):

    def test_trivial_reconcile(self):
        t = controldir.ControlDir.create_standalone_workingtree('.')
        out, err = self.run_bzr('reconcile')
        if t.branch.repository._reconcile_backsup_inventory:
            does_backup_text = 'Inventory ok.\n'
        else:
            does_backup_text = ''
        self.assertEqualDiff(out, 'Reconciling branch %s\nrevision_history ok.\nReconciling repository %s\n%sReconciliation complete.\n' % (t.branch.base, t.controldir.root_transport.base, does_backup_text))
        self.assertEqualDiff(err, '')

    def test_does_something_reconcile(self):
        t = controldir.ControlDir.create_standalone_workingtree('.')
        repo = t.branch.repository
        inv = inventory.Inventory(revision_id=b'missing')
        inv.root.revision = b'missing'
        repo.lock_write()
        with repo.lock_write(), WriteGroup(repo):
            repo.add_inventory(b'missing', inv, [])
        out, err = self.run_bzr('reconcile')
        if repo._reconcile_backsup_inventory:
            does_backup_text = 'Backup Inventory created.\nInventory regenerated.\n'
        else:
            does_backup_text = ''
        expected = 'Reconciling branch %s\nrevision_history ok.\nReconciling repository %s\n%sReconciliation complete.\n' % (t.branch.base, t.controldir.root_transport.base, does_backup_text)
        self.assertEqualDiff(expected, out)
        self.assertEqualDiff(err, '')