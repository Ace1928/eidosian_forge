import os
import stat
import sys
from breezy import tests
from breezy.bzr.branch import BzrBranch
from breezy.bzr.remote import RemoteBranchFormat
from breezy.controldir import ControlDir
from breezy.tests.test_permissions import check_mode_r
class TestPermissions(tests.TestCaseWithTransport):

    def test_new_branch(self):
        if isinstance(self.branch_format, RemoteBranchFormat):
            raise tests.TestNotApplicable('Remote branches have no permission logic')
        if sys.platform == 'win32':
            raise tests.TestNotApplicable('chmod has no effect on win32')
        os.mkdir('a')
        mode = stat.S_IMODE(os.stat('a').st_mode)
        t = self.make_branch_and_tree('.')
        if not isinstance(t.branch, BzrBranch):
            raise tests.TestNotApplicable('Only applicable to bzr branches')
        b = t.branch
        self.assertEqualMode(mode, b.controldir._get_dir_mode())
        self.assertEqualMode(mode & ~3657, b.controldir._get_file_mode())
        self.assertEqualMode(mode, b.control_files._dir_mode)
        self.assertEqualMode(mode & ~3657, b.control_files._file_mode)
        os.mkdir('d')
        os.chmod('d', 448)
        b = self.make_branch('d')
        self.assertEqualMode(448, b.controldir._get_dir_mode())
        self.assertEqualMode(384, b.controldir._get_file_mode())
        self.assertEqualMode(448, b.control_files._dir_mode)
        self.assertEqualMode(384, b.control_files._file_mode)
        check_mode_r(self, 'd/.bzr', 384, 448)

    def test_new_branch_group_sticky_bit(self):
        if isinstance(self.branch_format, RemoteBranchFormat):
            raise tests.TestNotApplicable('Remote branches have no permission logic')
        if sys.platform == 'win32':
            raise tests.TestNotApplicable('chmod has no effect on win32')
        elif sys.platform == 'darwin' or 'freebsd' in sys.platform:
            os.chown(self.test_dir, os.getuid(), os.getgid())
        t = self.make_branch_and_tree('.')
        b = t.branch
        if not isinstance(b, BzrBranch):
            raise tests.TestNotApplicable('Only applicable to bzr branches')
        os.mkdir('b')
        os.chmod('b', 1535)
        b = self.make_branch('b')
        self.assertEqualMode(1535, b.controldir._get_dir_mode())
        self.assertEqualMode(438, b.controldir._get_file_mode())
        self.assertEqualMode(1535, b.control_files._dir_mode)
        self.assertEqualMode(438, b.control_files._file_mode)
        check_mode_r(self, 'b/.bzr', 438, 1535)
        os.mkdir('c')
        os.chmod('c', 1512)
        b = self.make_branch('c')
        self.assertEqualMode(1512, b.controldir._get_dir_mode())
        self.assertEqualMode(416, b.controldir._get_file_mode())
        self.assertEqualMode(1512, b.control_files._dir_mode)
        self.assertEqualMode(416, b.control_files._file_mode)
        check_mode_r(self, 'c/.bzr', 416, 1512)

    def test_mode_0(self):
        """Test when a transport returns null permissions for .bzr"""
        if isinstance(self.branch_format, RemoteBranchFormat):
            raise tests.TestNotApplicable('Remote branches have no permission logic')
        self.make_branch_and_tree('.')
        bzrdir = ControlDir.open('.')
        _orig_stat = bzrdir.transport.stat

        def null_perms_stat(*args, **kwargs):
            result = _orig_stat(*args, **kwargs)
            return _NullPermsStat(result)
        bzrdir.transport.stat = null_perms_stat
        self.assertIs(None, bzrdir._get_dir_mode())
        self.assertIs(None, bzrdir._get_file_mode())