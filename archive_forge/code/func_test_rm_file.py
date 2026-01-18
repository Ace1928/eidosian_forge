from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_rm_file(self):
    self.run_script('$ echo content >file')
    self.assertPathExists('file')
    self.run_script('$ rm file')
    self.assertPathDoesNotExist('file')