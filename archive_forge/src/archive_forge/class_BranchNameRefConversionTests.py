from ... import tests
from ..object_store import BazaarObjectStore
from ..refs import BazaarRefsContainer, branch_name_to_ref, ref_to_branch_name
class BranchNameRefConversionTests(tests.TestCase):

    def test_head(self):
        self.assertEqual('', ref_to_branch_name(b'HEAD'))
        self.assertEqual(b'HEAD', branch_name_to_ref(''))

    def test_tag(self):
        self.assertRaises(ValueError, ref_to_branch_name, b'refs/tags/FOO')

    def test_branch(self):
        self.assertEqual('frost', ref_to_branch_name(b'refs/heads/frost'))
        self.assertEqual(b'refs/heads/frost', branch_name_to_ref('frost'))