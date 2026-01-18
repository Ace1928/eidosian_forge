from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def test_upgrade_standalone_branch(self):
    control = self.make_standalone_branch()
    tried, worked, issues = upgrade.smart_upgrade([control], format=self.to_format)
    self.assertLength(1, tried)
    self.assertEqual(tried[0], control)
    self.assertLength(1, worked)
    self.assertEqual(worked[0], control)
    self.assertLength(0, issues)
    self.assertPathExists('branch1/backup.bzr.~1~')
    self.assertEqual(control.open_repository()._format, self.to_format._repository_format)