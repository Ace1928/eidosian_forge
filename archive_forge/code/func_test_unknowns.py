from breezy.tests import TestCaseWithTransport
def test_unknowns(self):
    """Test that 'unknown' command reports unknown files"""
    tree = self.make_branch_and_tree('.')
    self.assertEqual(self.run_bzr('unknowns')[0], '')
    self.build_tree_contents([('a', b'contents of a\n')])
    self.assertEqual(self.run_bzr('unknowns')[0], 'a\n')
    self.build_tree(['b', 'c', 'd e'])
    self.assertEqual(self.run_bzr('unknowns')[0], 'a\nb\nc\n"d e"\n')
    tree.add(['a', 'd e'])
    self.assertEqual(self.run_bzr('unknowns')[0], 'b\nc\n')
    tree.add(['b', 'c'])
    self.assertEqual(self.run_bzr('unknowns')[0], '')