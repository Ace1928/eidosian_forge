import os
import sys
from .. import bedding, osutils, tests
class TestXDGConfigDir(tests.TestCaseInTempDir):

    def setUp(self):
        if sys.platform == 'win32':
            raise tests.TestNotApplicable('XDG config dir not used on this platform')
        super().setUp()
        self.overrideEnv('HOME', self.test_home_dir)
        self.overrideEnv('BRZ_HOME', None)

    def test_xdg_config_dir_exists(self):
        """When ~/.config/bazaar exists, use it as the config dir."""
        newdir = osutils.pathjoin(self.test_home_dir, '.config', 'bazaar')
        os.makedirs(newdir)
        self.assertEqual(bedding.config_dir(), newdir)

    def test_xdg_config_home(self):
        """When XDG_CONFIG_HOME is set, use it."""
        xdgconfigdir = osutils.pathjoin(self.test_home_dir, 'xdgconfig')
        self.overrideEnv('XDG_CONFIG_HOME', xdgconfigdir)
        newdir = osutils.pathjoin(xdgconfigdir, 'bazaar')
        os.makedirs(newdir)
        self.assertEqual(bedding.config_dir(), newdir)

    def test_ensure_config_dir_exists(self):
        xdgconfigdir = osutils.pathjoin(self.test_home_dir, 'xdgconfig')
        self.overrideEnv('XDG_CONFIG_HOME', xdgconfigdir)
        bedding.ensure_config_dir_exists()
        newdir = osutils.pathjoin(xdgconfigdir, 'breezy')
        self.assertTrue(os.path.isdir(newdir))