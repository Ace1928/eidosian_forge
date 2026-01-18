from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_null_output_matches_option(self):
    """If you want null output to be a wild card, you can pass
        null_output_matches_anything to run_script"""
    self.run_script('\n            $ echo foo\n            ', null_output_matches_anything=True)