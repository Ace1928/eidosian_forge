from breezy import gpg, tests
def test_verify_signatures_verbose(self):
    wt = self.setup_tree()
    self.monkey_patch_gpg()
    self.run_bzr('sign-my-commits')
    out = self.run_bzr('verify-signatures --verbose', retcode=1)
    self.assertEqual(('4 commits with valid signatures\n  None signed 4 commits\n0 commits with key now expired\n0 commits with unknown keys\n0 commits not valid\n1 commit not signed\n  1 commit by author Alternate <alt@foo.com>\n', ''), out)