import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_submit_branch(self):
    self.prepare_merge_directive()
    self.run_bzr_error(('No submit branch',), 'merge-directive', retcode=3)
    self.run_bzr('merge-directive ../tree2')