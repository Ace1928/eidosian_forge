import os
import sys
from .. import bedding, osutils, tests
def test_xdg_config_dir_exists(self):
    """When ~/.config/bazaar exists, use it as the config dir."""
    newdir = osutils.pathjoin(self.test_home_dir, '.config', 'bazaar')
    os.makedirs(newdir)
    self.assertEqual(bedding.config_dir(), newdir)