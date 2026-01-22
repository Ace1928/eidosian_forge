from breezy import errors, ui
from breezy.tests.per_repository_reference import \
class TestBreakLock(TestCaseWithExternalReferenceRepository):

    def test_break_lock(self):
        base = self.make_repository('base')
        repo = self.make_referring('referring', base)
        unused_repo = repo.controldir.open_repository()
        base.lock_write()
        self.addCleanup(base.unlock)
        repo.lock_write()
        self.assertEqual(repo.get_physical_lock_status(), unused_repo.get_physical_lock_status())
        if not unused_repo.get_physical_lock_status():
            repo.unlock()
            return
        ui.ui_factory = ui.CannedInputUIFactory([True])
        unused_repo.break_lock()
        self.assertRaises(errors.LockBroken, repo.unlock)