import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_on_previous_rev(self):
    """switch to previous rev in a standalone directory

        Inspired by: https://bugs.launchpad.net/brz/+bug/1018628
        """
    self.script_runner = script.ScriptRunner()
    self.script_runner.run_script(self, '\n           $ brz init\n           Created a standalone tree (format: 2a)\n           $ brz commit -m 1 --unchanged\n           $ brz commit -m 2 --unchanged\n           $ brz switch -r 1\n           2>brz: ERROR: switching would create a branch reference loop. Use the "bzr up" command to switch to a different revision.', null_output_matches_anything=True)