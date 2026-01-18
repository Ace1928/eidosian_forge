import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_rename_symlink(self):
    self.make_branch_and_working_tree()
    old_name, new_name = ('old-link', 'new-link')
    self.add_symlink(old_name, 'target')
    self.do_full_upload()
    self.rename_any(old_name, new_name)
    self.do_upload()
    self.assertUpPathExists(new_name)