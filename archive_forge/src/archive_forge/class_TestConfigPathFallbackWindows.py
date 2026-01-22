import os
import sys
from .. import bedding, osutils, tests
class TestConfigPathFallbackWindows(tests.TestCaseInTempDir):

    def mock_special_folder_path(self, csidl):
        if csidl == win32utils.CSIDL_APPDATA:
            return self.appdata
        elif csidl == win32utils.CSIDL_PERSONAL:
            return self.test_dir
        return None

    def setUp(self):
        if sys.platform != 'win32':
            raise tests.TestNotApplicable('This test is specific to Windows platform')
        super().setUp()
        self.appdata = os.path.join(self.test_dir, 'appdata')
        self.appdata_bzr = os.path.join(self.appdata, 'bazaar', '2.0')
        os.makedirs(self.appdata_bzr)
        self.overrideAttr(win32utils, '_get_sh_special_folder_path', self.mock_special_folder_path)
        self.overrideEnv('BRZ_HOME', None)
        self.overrideEnv('BZR_HOME', None)

    def test_config_dir(self):
        self.assertIsSameRealPath(bedding.config_dir(), self.appdata_bzr)

    def test_config_dir_is_unicode(self):
        self.assertIsInstance(bedding.config_dir(), str)

    def test_config_path(self):
        self.assertIsSameRealPath(bedding.config_path(), self.appdata_bzr + '/bazaar.conf')
        self.overrideAttr(win32utils, 'get_appdata_location', lambda: None)
        self.assertRaises(RuntimeError, bedding.config_path)

    def test_locations_config_path(self):
        self.assertIsSameRealPath(bedding.locations_config_path(), self.appdata_bzr + '/locations.conf')
        self.overrideAttr(win32utils, 'get_appdata_location', lambda: None)
        self.assertRaises(RuntimeError, bedding.locations_config_path)

    def test_authentication_config_path(self):
        self.assertIsSameRealPath(bedding.authentication_config_path(), self.appdata_bzr + '/authentication.conf')
        self.overrideAttr(win32utils, 'get_appdata_location', lambda: None)
        self.assertRaises(RuntimeError, bedding.authentication_config_path)