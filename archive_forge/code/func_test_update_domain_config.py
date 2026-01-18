import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_update_domain_config(self):
    config = fixtures.DomainConfig(self.client, self.test_domain.id)
    self.useFixture(config)
    update_config_ref = self._new_ref()
    config_ret = self.client.domain_configs.update(self.test_domain.id, update_config_ref)
    self.check_domain_config(config_ret, update_config_ref)