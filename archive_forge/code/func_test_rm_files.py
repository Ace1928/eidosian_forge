from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_rm_files(self):
    self.run_script('\n$ echo content >file\n$ echo content >file2\n')
    self.assertPathExists('file2')
    self.run_script('$ rm file file2')
    self.assertPathDoesNotExist('file2')