from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_two_args_sets(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    tree.add('file')
    out, err = self.run_bzr('reference -d tree file http://example.org')
    location = tree.get_reference_info('file')
    self.assertEqual('http://example.org', location)
    self.assertEqual('', out)
    self.assertEqual('', err)