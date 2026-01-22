import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
class TestWithUpgradableBranches(TestCaseWithTransport):

    def setUp(self):
        super().setUp()

    def make_current_format_branch_and_checkout(self):
        current_tree = self.make_branch_and_tree('current_format_branch', format='default')
        current_tree.branch.create_checkout(self.get_url('current_format_checkout'), lightweight=True)

    def test_readonly_url_error(self):
        self.make_branch_and_tree('old_format_branch', format='knit')
        out, err = self.run_bzr(['upgrade', self.get_readonly_url('old_format_branch')], retcode=3)
        err_msg = 'Upgrade URL cannot work with readonly URLs.'
        self.assertEqualDiff('conversion error: %s\nbrz: ERROR: %s\n' % (err_msg, err_msg), err)

    def test_upgrade_up_to_date(self):
        self.make_current_format_branch_and_checkout()
        burl = self.get_transport('current_format_branch').local_abspath('.')
        out, err = self.run_bzr('upgrade current_format_branch', retcode=0)
        self.assertEqual('Upgrading branch %s/ ...\nThe branch format %s is already at the most recent format.\n' % (burl, 'Meta directory format 1'), out)

    def test_upgrade_up_to_date_checkout_warns_branch_left_alone(self):
        self.make_current_format_branch_and_checkout()
        burl = self.get_transport('current_format_branch').local_abspath('.')
        curl = self.get_transport('current_format_checkout').local_abspath('.')
        out, err = self.run_bzr('upgrade current_format_checkout', retcode=0)
        self.assertEqual('Upgrading branch %s/ ...\nThis is a checkout. The branch (%s/) needs to be upgraded separately.\nThe branch format %s is already at the most recent format.\n' % (curl, burl, 'Meta directory format 1'), out)

    def test_upgrade_checkout(self):
        pass

    def test_upgrade_repository_scans_branches(self):
        pass

    def test_upgrade_branch_in_repo(self):
        pass

    def test_upgrade_control_dir(self):
        old_format = OldBzrDirFormat()
        self.addCleanup(bzr.BzrProber.formats.remove, old_format.get_format_string())
        bzr.BzrProber.formats.register(old_format.get_format_string(), old_format)
        self.addCleanup(controldir.ControlDirFormat._set_default_format, controldir.ControlDirFormat.get_default_format())
        path = 'old_format_branch'
        self.make_branch_and_tree(path, format=old_format)
        transport = self.get_transport(path)
        url = transport.base
        display_url = transport.local_abspath('.')
        controldir.ControlDirFormat._set_default_format(old_format)
        backup_dir = 'backup.bzr.~1~'
        out, err = self.run_bzr(['upgrade', '--format=2a', url])
        self.assertEqualDiff('Upgrading branch {}/ ...\nstarting upgrade of {}/\nmaking backup of {}/.bzr\n  to {}/{}\nstarting upgrade from old test format to 2a\nfinished\n'.format(display_url, display_url, display_url, display_url, backup_dir), out)
        self.assertEqualDiff('', err)
        self.assertTrue(isinstance(controldir.ControlDir.open(self.get_url(path))._format, bzrdir.BzrDirMetaFormat1))

    def test_upgrade_explicit_knit(self):
        self.make_branch_and_tree('branch', format='knit')
        transport = self.get_transport('branch')
        url = transport.base
        display_url = transport.local_abspath('.')
        backup_dir = 'backup.bzr.~1~'
        out, err = self.run_bzr(['upgrade', '--format=pack-0.92', url])
        self.assertEqualDiff('Upgrading branch {}/ ...\nstarting upgrade of {}/\nmaking backup of {}/.bzr\n  to {}/{}\nstarting repository conversion\nrepository converted\nfinished\n'.format(display_url, display_url, display_url, display_url, backup_dir), out)
        self.assertEqualDiff('', err)
        converted_dir = controldir.ControlDir.open(self.get_url('branch'))
        self.assertTrue(isinstance(converted_dir._format, bzrdir.BzrDirMetaFormat1))
        self.assertTrue(isinstance(converted_dir.open_repository()._format, RepositoryFormatKnitPack1))

    def test_upgrade_repo(self):
        self.run_bzr('init-shared-repository --format=pack-0.92 repo')
        self.run_bzr('upgrade --format=2a repo')

    def assertLegalOption(self, option_str):
        self.run_bzr('init --format=pack-0.92 branch-foo')
        self.run_bzr('upgrade --format=2a branch-foo {}'.format(option_str))

    def assertBranchFormat(self, dir, format):
        branch = controldir.ControlDir.open_tree_or_branch(self.get_url(dir))[1]
        branch_format = branch._format
        meta_format = controldir.format_registry.make_controldir(format)
        expected_format = meta_format.get_branch_format()
        self.assertEqual(expected_format, branch_format)

    def test_upgrade_clean_supported(self):
        self.assertLegalOption('--clean')
        self.assertBranchFormat('branch-foo', '2a')
        backup_bzr_dir = os.path.join('branch-foo', 'backup.bzr.~1~')
        self.assertFalse(os.path.exists(backup_bzr_dir))

    def test_upgrade_dry_run_supported(self):
        self.assertLegalOption('--dry-run')
        self.assertBranchFormat('branch-foo', 'pack-0.92')

    def test_upgrade_permission_check(self):
        """'backup.bzr' should retain permissions of .bzr. Bug #262450"""
        self.requireFeature(features.posix_permissions_feature)
        old_perms = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        backup_dir = 'backup.bzr.~1~'
        self.run_bzr('init --format=1.6')
        os.chmod('.bzr', old_perms)
        self.run_bzr('upgrade')
        new_perms = os.stat(backup_dir).st_mode & 511
        self.assertTrue(new_perms == old_perms)

    def test_upgrade_with_existing_backup_dir(self):
        self.make_branch_and_tree('old_format_branch', format='knit')
        t = self.get_transport('old_format_branch')
        url = t.base
        display_url = t.local_abspath('.')
        backup_dir1 = 'backup.bzr.~1~'
        backup_dir2 = 'backup.bzr.~2~'
        t.mkdir(backup_dir1)
        out, err = self.run_bzr(['upgrade', '--format=2a', url])
        self.assertEqualDiff('Upgrading branch {}/ ...\nstarting upgrade of {}/\nmaking backup of {}/.bzr\n  to {}/{}\nstarting repository conversion\nrepository converted\nfinished\n'.format(display_url, display_url, display_url, display_url, backup_dir2), out)
        self.assertEqualDiff('', err)
        self.assertTrue(isinstance(controldir.ControlDir.open(self.get_url('old_format_branch'))._format, bzrdir.BzrDirMetaFormat1))
        self.assertTrue(t.has(backup_dir2))