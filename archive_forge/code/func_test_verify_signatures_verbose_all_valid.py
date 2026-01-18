from breezy import gpg, tests
def test_verify_signatures_verbose_all_valid(self):
    wt = self.setup_tree()
    self.monkey_patch_gpg()
    self.run_bzr('sign-my-commits')
    self.run_bzr(['sign-my-commits', '.', 'Alternate <alt@foo.com>'])
    out = self.run_bzr('verify-signatures --verbose')
    self.assertEqual(('All commits signed with verifiable keys\n  None signed 5 commits\n', ''), out)