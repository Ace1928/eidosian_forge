import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_config(self):
    """Call ``PATCH /domains/{domain_id}/config``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    new_config = {'ldap': {'url': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
    r = self.patch('/domains/%(domain_id)s/config' % {'domain_id': self.domain['id']}, body={'config': new_config})
    res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
    expected_config = copy.deepcopy(self.config)
    expected_config['ldap']['url'] = new_config['ldap']['url']
    expected_config['identity']['driver'] = new_config['identity']['driver']
    self.assertEqual(expected_config, r.result['config'])
    self.assertEqual(expected_config, res)