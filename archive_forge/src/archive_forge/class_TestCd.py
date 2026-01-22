from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestCd(script.TestCaseWithTransportAndScript):

    def test_cd_usage(self):
        self.assertRaises(SyntaxError, self.run_script, '$ cd foo bar')

    def test_cd_out_of_jail(self):
        self.assertRaises(ValueError, self.run_script, '$ cd /out-of-jail')
        self.assertRaises(ValueError, self.run_script, '$ cd ..')

    def test_cd_dir_and_back_home(self):
        self.assertEqual(self.test_dir, osutils.getcwd())
        self.run_script('\n$ mkdir dir\n$ cd dir\n')
        self.assertEqual(osutils.pathjoin(self.test_dir, 'dir'), osutils.getcwd())
        self.run_script('$ cd')
        self.assertEqual(self.test_dir, osutils.getcwd())