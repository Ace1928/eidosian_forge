import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_config_invalid_group(self):
    """Call ``PATCH /domains/{domain_id}/config/{invalid_group}``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    invalid_group = uuid.uuid4().hex
    new_config = {invalid_group: {'url': uuid.uuid4().hex, 'user_filter': uuid.uuid4().hex}}
    self.patch('/domains/%(domain_id)s/config/%(invalid_group)s' % {'domain_id': self.domain['id'], 'invalid_group': invalid_group}, body={'config': new_config}, expected_status=http.client.FORBIDDEN)
    config = {'ldap': {'suffix': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    new_config = {'identity': {'driver': uuid.uuid4().hex}}
    self.patch('/domains/%(domain_id)s/config/identity' % {'domain_id': self.domain['id']}, body={'config': new_config}, expected_status=http.client.NOT_FOUND)