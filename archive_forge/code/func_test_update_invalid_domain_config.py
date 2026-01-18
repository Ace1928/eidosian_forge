import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_update_invalid_domain_config(self):
    config = fixtures.DomainConfig(self.client, self.test_domain.id)
    self.useFixture(config)
    invalid_groups_ref = {uuid.uuid4().hex: {uuid.uuid4().hex: uuid.uuid4().hex}, uuid.uuid4().hex: {uuid.uuid4().hex: uuid.uuid4().hex}}
    self.assertRaises(http.Forbidden, self.client.domain_configs.update, self.test_domain.id, invalid_groups_ref)
    invalid_options_ref = {'identity': {uuid.uuid4().hex: uuid.uuid4().hex}, 'ldap': {uuid.uuid4().hex: uuid.uuid4().hex}}
    self.assertRaises(http.Forbidden, self.client.domain_configs.update, self.test_domain.id, invalid_options_ref)