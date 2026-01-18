import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_token_revoked_once_domain_disabled(self):
    """Test token from a disabled domain has been invalidated.

        Test that a token that was valid for an enabled domain
        becomes invalid once that domain is disabled.

        """
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user2 = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
    auth_body = self.build_authentication_request(user_id=user2['id'], password=user2['password'])
    token_resp = self.post('/auth/tokens', body=auth_body)
    subject_token = token_resp.headers.get('x-subject-token')
    self.head('/auth/tokens', headers={'x-subject-token': subject_token}, expected_status=http.client.OK)
    domain['enabled'] = False
    url = '/domains/%(domain_id)s' % {'domain_id': domain['id']}
    self.patch(url, body={'domain': {'enabled': False}})
    self.head('/auth/tokens', headers={'x-subject-token': subject_token}, expected_status=http.client.NOT_FOUND)