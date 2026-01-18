import testtools
from glanceclient.v1.apiclient import base
def test_two_resources_with_diff_id_are_not_equal(self):
    r1 = base.Resource(None, {'id': 1, 'name': 'hi'})
    r2 = base.Resource(None, {'id': 2, 'name': 'hello'})
    self.assertNotEqual(r1, r2)