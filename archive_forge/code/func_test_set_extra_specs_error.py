from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import type
from openstack import exceptions
from openstack.tests.unit import base
def test_set_extra_specs_error(self):
    sess = mock.Mock()
    response = mock.Mock()
    response.status_code = 400
    response.content = None
    sess.post.return_value = response
    sot = type.Type(id=FAKE_ID)
    set_specs = {'lol': 'rofl'}
    self.assertRaises(exceptions.BadRequestException, sot.set_extra_specs, sess, **set_specs)