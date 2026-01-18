from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_one_arg_uses_containing_tree(self):
    tree = self.make_tree_with_reference()
    out, err = self.run_bzr('reference -d tree newpath')
    self.assertEqual('newpath http://example.org\n', out)