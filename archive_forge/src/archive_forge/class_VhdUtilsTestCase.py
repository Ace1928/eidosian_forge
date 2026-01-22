import os
import tempfile
from os_win import constants
from os_win.tests.functional import test_base
from os_win import utilsfactory
class VhdUtilsTestCase(test_base.OsWinBaseFunctionalTestCase):

    def setUp(self):
        super(VhdUtilsTestCase, self).setUp()
        self._vhdutils = utilsfactory.get_vhdutils()
        self._diskutils = utilsfactory.get_diskutils()
        self._pathutils = utilsfactory.get_pathutils()

    def _create_temp_vhd(self, size_mb=32, vhd_type=constants.VHD_TYPE_DYNAMIC):
        f = tempfile.TemporaryFile(suffix='.vhdx', prefix='oswin_vhdtest_')
        f.close()
        self._vhdutils.create_vhd(f.name, vhd_type, max_internal_size=size_mb << 20)
        self.addCleanup(os.unlink, f.name)
        return f.name

    def _create_temp_symlink(self, target, target_is_dir):
        f = tempfile.TemporaryFile(prefix='oswin_vhdtest_link_')
        f.close()
        self._pathutils.create_sym_link(f.name, target, target_is_dir)
        if target_is_dir:
            self.addCleanup(os.rmdir, f.name)
        else:
            self.addCleanup(os.unlink, f.name)
        return f.name

    def test_attach_detach(self):
        vhd_path = self._create_temp_vhd()
        vhd_link = self._create_temp_symlink(vhd_path, target_is_dir=False)
        vhd_dir_link = self._create_temp_symlink(os.path.dirname(vhd_path), target_is_dir=True)
        vhd_link2 = os.path.join(vhd_dir_link, os.path.basename(vhd_path))

        def _check_attached(expect_attached):
            paths = [vhd_path, vhd_link, vhd_link2]
            for path in paths:
                self.assertEqual(expect_attached, self._vhdutils.is_virtual_disk_file_attached(path))
                self.assertEqual(expect_attached, self._diskutils.is_virtual_disk_file_attached(path))
        _check_attached(False)
        try:
            self._vhdutils.attach_virtual_disk(vhd_path)
            _check_attached(True)
        finally:
            self._vhdutils.detach_virtual_disk(vhd_path)
            _check_attached(False)