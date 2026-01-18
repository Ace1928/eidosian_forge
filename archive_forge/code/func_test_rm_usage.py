from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_rm_usage(self):
    self.assertRaises(SyntaxError, self.run_script, '$ rm')
    self.assertRaises(SyntaxError, self.run_script, '$ rm -ff foo')