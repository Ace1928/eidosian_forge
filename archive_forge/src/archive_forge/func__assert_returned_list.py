from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def _assert_returned_list(self, ref_list, returned_list):
    self.assertEqual(len(ref_list), len(returned_list))
    [self.assertIsInstance(r, self.model) for r in returned_list]