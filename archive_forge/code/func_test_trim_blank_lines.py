from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_trim_blank_lines(self):
    """Blank lines are respected, but trimmed at the start and end.

        Python triple-quoted syntax is going to give stubby/empty blank lines
        right at the start and the end.  These are cut off so that callers don't
        need special syntax to avoid them.

        However we do want to be able to match commands that emit blank lines.
        """
    self.assertEqual([(['bar'], None, '\n', None)], script._script_to_commands('\n            $bar\n\n            '))