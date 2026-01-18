from breezy.tests import TestCaseWithTransport
def test_unknowns_directory(self):
    """Test --directory option"""
    tree = self.make_branch_and_tree('a')
    self.build_tree(['a/README'])
    out, err = self.run_bzr(['unknowns', '--directory=a'])
    self.assertEqual('README\n', out)