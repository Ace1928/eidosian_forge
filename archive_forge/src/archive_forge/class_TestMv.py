from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestMv(script.TestCaseWithTransportAndScript):

    def test_usage(self):
        self.assertRaises(SyntaxError, self.run_script, '$ mv')
        self.assertRaises(SyntaxError, self.run_script, '$ mv f')
        self.assertRaises(SyntaxError, self.run_script, '$ mv f1 f2 f3')

    def test_move_file(self):
        self.run_script('$ echo content >file')
        self.assertPathExists('file')
        self.run_script('$ mv file new_name')
        self.assertPathDoesNotExist('file')
        self.assertPathExists('new_name')

    def test_move_unknown_file(self):
        self.assertRaises(AssertionError, self.run_script, '$ mv unknown does-not-exist')

    def test_move_dir(self):
        self.run_script('\n$ mkdir dir\n$ echo content >dir/file\n')
        self.run_script('$ mv dir new_name')
        self.assertPathDoesNotExist('dir')
        self.assertPathExists('new_name')
        self.assertPathExists('new_name/file')

    def test_move_file_into_dir(self):
        self.run_script('\n$ mkdir dir\n$ echo content > file\n')
        self.run_script('$ mv file dir')
        self.assertPathExists('dir')
        self.assertPathDoesNotExist('file')
        self.assertPathExists('dir/file')