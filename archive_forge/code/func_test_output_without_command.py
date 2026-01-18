from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_output_without_command(self):
    self.assertRaises(SyntaxError, script._script_to_commands, '>input')