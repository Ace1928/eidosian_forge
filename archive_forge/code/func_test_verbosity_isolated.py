from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_verbosity_isolated(self):
    """Global verbosity is isolated from commands run in scripts.
        """
    self.run_script('\n        $ brz init --quiet a\n        ')
    self.assertEqual(trace.is_quiet(), False)