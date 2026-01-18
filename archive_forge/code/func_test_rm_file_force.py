from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_rm_file_force(self):
    self.assertPathDoesNotExist('file')
    self.run_script('$ rm -f file')
    self.assertPathDoesNotExist('file')