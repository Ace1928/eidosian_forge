import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
class TestSwitchStandAloneCorruption(TestCaseWithTransport):

    def test_empty_tree_switch(self):
        """switch . on an empty tree gets infinite recursion

        Inspired by: https://bugs.launchpad.net/bzr/+bug/1018628
        """
        self.script_runner = script.ScriptRunner()
        self.script_runner.run_script(self, '\n            $ brz init\n            Created a standalone tree (format: 2a)\n            $ brz switch .\n            2>brz: ERROR: switching would create a branch reference loop. Use the "bzr up" command to switch to a different revision.\n            ')

    def test_switch_on_previous_rev(self):
        """switch to previous rev in a standalone directory

        Inspired by: https://bugs.launchpad.net/brz/+bug/1018628
        """
        self.script_runner = script.ScriptRunner()
        self.script_runner.run_script(self, '\n           $ brz init\n           Created a standalone tree (format: 2a)\n           $ brz commit -m 1 --unchanged\n           $ brz commit -m 2 --unchanged\n           $ brz switch -r 1\n           2>brz: ERROR: switching would create a branch reference loop. Use the "bzr up" command to switch to a different revision.', null_output_matches_anything=True)

    def test_switch_create_colo_locks_repo_path(self):
        self.script_runner = script.ScriptRunner()
        self.script_runner.run_script(self, '\n            $ mkdir mywork\n            $ cd mywork\n            $ brz init\n            Created a standalone tree (format: 2a)\n            $ echo A > a && brz add a && brz commit -m A\n            $ brz switch -b br1\n            $ cd ..\n            $ mv mywork mywork1\n            $ cd mywork1\n            $ brz branches\n              br1\n            ', null_output_matches_anything=True)

    def test_switch_to_new_branch_on_old_rev(self):
        """switch to previous rev in a standalone directory

        Inspired by: https://bugs.launchpad.net/brz/+bug/933362
        """
        self.script_runner = script.ScriptRunner()
        self.script_runner.run_script(self, '\n           $ brz init\n           Created a standalone tree (format: 2a)\n           $ brz switch -b trunk\n           2>Tree is up to date at revision 0.\n           2>Switched to branch trunk\n           $ brz commit -m 1 --unchanged\n           2>Committing to: ...\n           2>Committed revision 1.\n           $ brz commit -m 2 --unchanged\n           2>Committing to: ...\n           2>Committed revision 2.\n           $ brz switch -b blah -r1\n           2>Updated to revision 1.\n           2>Switched to branch blah\n           $ brz branches\n           * blah\n             trunk\n           $ brz st\n           ')