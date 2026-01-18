import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def test_patch_types(self):
    self.prepare_merge_directive()
    md_text = self.run_bzr('merge-directive ../tree2')[0]
    self.assertContainsRe(md_text, '# Begin bundle')
    self.assertContainsRe(md_text, '\\+e')
    md_text = self.run_bzr('merge-directive ../tree2 --diff .')[0]
    self.assertNotContainsRe(md_text, '# Begin bundle')
    self.assertContainsRe(md_text, '\\+e')
    md_text = self.run_bzr('merge-directive --plain')[0]
    self.assertNotContainsRe(md_text, '\\+e')