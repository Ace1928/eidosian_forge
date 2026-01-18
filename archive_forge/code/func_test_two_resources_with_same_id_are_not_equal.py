from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_two_resources_with_same_id_are_not_equal(self):
    r1 = base.Resource(None, {'id': 1, 'name': 'hi'})
    r2 = base.Resource(None, {'id': 1, 'name': 'hello'})
    self.assertNotEqual(r1, r2)