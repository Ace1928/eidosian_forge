from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_two_resources_with_no_id_are_equal(self):
    r1 = base.Resource(None, {'name': 'joe', 'age': 12})
    r2 = base.Resource(None, {'name': 'joe', 'age': 12})
    self.assertEqual(r1, r2)