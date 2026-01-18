import os
from breezy import tests
def test_pack_clean_obsolete_packs(self):
    """Ensure --clean-obsolete-packs removes obsolete pack files
        """
    wt = self.make_branch_and_tree('.')
    t = wt.branch.repository.controldir.transport
    self._make_versioned_file('file0.txt')
    for i in range(5):
        self._update_file('file0.txt', 'HELLO %d\n' % i)
    out, err = self.run_bzr(['pack', '--clean-obsolete-packs'])
    pack_names = t.list_dir('repository/obsolete_packs')
    self.assertTrue(len(pack_names) == 0)