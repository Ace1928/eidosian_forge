from breezy import conflicts, tests, workingtree
from breezy.tests import features, script
def test_text_conflict_paths(self):
    """Text conflicts on non-ascii paths are displayed okay"""
    make_tree_with_conflicts(self, 'branch', prefix='§')
    out, err = self.run_bzr(['conflicts', '-d', 'branch', '--text'], encoding=self.encoding)
    self.assertEqual(out, '§_other_file\n§file\n')
    self.assertEqual(err, '')