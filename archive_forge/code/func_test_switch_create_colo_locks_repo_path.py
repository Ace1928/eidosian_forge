import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_create_colo_locks_repo_path(self):
    self.script_runner = script.ScriptRunner()
    self.script_runner.run_script(self, '\n            $ mkdir mywork\n            $ cd mywork\n            $ brz init\n            Created a standalone tree (format: 2a)\n            $ echo A > a && brz add a && brz commit -m A\n            $ brz switch -b br1\n            $ cd ..\n            $ mv mywork mywork1\n            $ cd mywork1\n            $ brz branches\n              br1\n            ', null_output_matches_anything=True)