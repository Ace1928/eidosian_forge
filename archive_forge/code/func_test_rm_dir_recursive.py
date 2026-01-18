from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_rm_dir_recursive(self):
    self.run_script('\n$ mkdir dir\n$ rm -r dir\n')
    self.assertPathDoesNotExist('dir')