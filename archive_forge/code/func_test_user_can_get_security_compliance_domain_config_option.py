import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_security_compliance_domain_config_option(self):
    password_regex_description = uuid.uuid4().hex
    self.config_fixture.config(group='security_compliance', password_regex_description=password_regex_description)
    with self.test_client() as c:
        c.get('/v3/domains/%s/config/security_compliance/password_regex_description' % CONF.identity.default_domain_id, headers=self.headers)