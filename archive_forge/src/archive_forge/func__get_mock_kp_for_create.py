import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _get_mock_kp_for_create(self, key_name, public_key=None, priv_saved=False, key_type=None, user=None):
    template = copy.deepcopy(self.kp_template)
    template['resources']['kp']['properties']['name'] = key_name
    props = template['resources']['kp']['properties']
    if public_key:
        props['public_key'] = public_key
    gen_pk = public_key or 'generated test public key'
    nova_key = self._mock_key(key_name, gen_pk)
    if priv_saved:
        nova_key.private_key = 'private key for %s' % key_name
        props['save_private_key'] = True
    if key_type:
        props['type'] = key_type
    if user:
        props['user'] = user
    kp_res = self._get_test_resource(template)
    self.patchobject(self.fake_keypairs, 'create', return_value=nova_key)
    return (kp_res, nova_key)