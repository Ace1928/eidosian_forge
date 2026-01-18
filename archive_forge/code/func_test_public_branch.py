import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_public_branch(self):
    self.prepare_merge_directive()
    self.run_bzr_error(('No public branch',), 'merge-directive --diff ../tree2', retcode=3)
    md_text = self.run_bzr('merge-directive ../tree2')[0]
    self.assertNotContainsRe(md_text, 'source_branch:')
    self.run_bzr('merge-directive --diff ../tree2 .')
    self.run_bzr('merge-directive --diff')[0]
    self.assertNotContainsRe(md_text, 'source_branch:')