from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import type
from openstack import exceptions
from openstack.tests.unit import base
def test_set_extra_specs(self):
    response = mock.Mock()
    response.status_code = 200
    response.json.return_value = self.extra_specs_result
    sess = mock.Mock()
    sess.post.return_value = response
    sot = type.Type(id=FAKE_ID)
    set_specs = {'lol': 'rofl'}
    result = sot.set_extra_specs(sess, **set_specs)
    self.assertEqual(result, self.extra_specs_result['extra_specs'])
    sess.post.assert_called_once_with('types/' + FAKE_ID + '/extra_specs', headers={}, json={'extra_specs': set_specs})