from breezy import gpg, tests
def test_sign_diff_committer(self):
    wt = self.setup_tree()
    repo = wt.branch.repository
    self.monkey_patch_gpg()
    self.run_bzr(['sign-my-commits', '.', 'Alternate <alt@foo.com>'])
    self.assertUnsigned(repo, b'A')
    self.assertUnsigned(repo, b'B')
    self.assertUnsigned(repo, b'C')
    self.assertSigned(repo, b'D')