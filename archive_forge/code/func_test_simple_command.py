from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_simple_command(self):
    self.assertEqual([(['cd', 'trunk'], None, None, None)], script._script_to_commands('$ cd trunk'))