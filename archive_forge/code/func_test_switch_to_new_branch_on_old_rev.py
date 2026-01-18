import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_to_new_branch_on_old_rev(self):
    """switch to previous rev in a standalone directory

        Inspired by: https://bugs.launchpad.net/brz/+bug/933362
        """
    self.script_runner = script.ScriptRunner()
    self.script_runner.run_script(self, '\n           $ brz init\n           Created a standalone tree (format: 2a)\n           $ brz switch -b trunk\n           2>Tree is up to date at revision 0.\n           2>Switched to branch trunk\n           $ brz commit -m 1 --unchanged\n           2>Committing to: ...\n           2>Committed revision 1.\n           $ brz commit -m 2 --unchanged\n           2>Committing to: ...\n           2>Committed revision 2.\n           $ brz switch -b blah -r1\n           2>Updated to revision 1.\n           2>Switched to branch blah\n           $ brz branches\n           * blah\n             trunk\n           $ brz st\n           ')