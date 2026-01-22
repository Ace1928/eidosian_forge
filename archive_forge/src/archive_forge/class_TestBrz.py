from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestBrz(script.TestCaseWithTransportAndScript):

    def test_brz_smoke(self):
        self.run_script('\n            $ brz init branch\n            Created a standalone tree (format: ...)\n            ')
        self.assertPathExists('branch')