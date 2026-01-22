import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def _checkout_and_switch(self, option=''):
    self.script_runner.run_script(self, '\n                $ brz checkout %(option)s repo/trunk checkout\n                $ cd checkout\n                $ brz switch --create-branch switched\n                2>Tree is up to date at revision 0.\n                2>Switched to branch at .../switched/\n                $ cd ..\n                ' % locals())
    bound_branch = branch.Branch.open_containing('checkout')[0]
    master_branch = branch.Branch.open_containing('repo/switched')[0]
    return (bound_branch, master_branch)