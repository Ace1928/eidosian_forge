import testtools
from glanceclient.v1.apiclient import base
def test_two_resources_with_not_eq_info_are_not_equal(self):
    r1 = base.Resource(None, {'name': 'bill', 'age': 21})
    r2 = base.Resource(None, {'name': 'joe', 'age': 12})
    self.assertNotEqual(r1, r2)