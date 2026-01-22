import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _UserAccessRuleTests(object):
    """Test cases for anyone that has a valid user token."""

    def test_user_can_get_their_access_rules(self):
        access_rule_id = uuid.uuid4().hex
        app_cred = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'user_id': self.user_id, 'project_id': self.project_id, 'secret': uuid.uuid4().hex, 'access_rules': [{'id': access_rule_id, 'service': uuid.uuid4().hex, 'path': uuid.uuid4().hex, 'method': uuid.uuid4().hex[16:]}]}
        PROVIDERS.application_credential_api.create_application_credential(app_cred)
        with self.test_client() as c:
            path = '/v3/users/%s/access_rules/%s' % (self.user_id, app_cred['access_rules'][0]['id'])
            c.get(path, headers=self.headers)

    def test_user_can_list_their_access_rules(self):
        app_cred = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'user_id': self.user_id, 'project_id': self.project_id, 'secret': uuid.uuid4().hex, 'access_rules': [{'id': uuid.uuid4().hex, 'service': uuid.uuid4().hex, 'path': uuid.uuid4().hex, 'method': uuid.uuid4().hex[16:]}]}
        PROVIDERS.application_credential_api.create_application_credential(app_cred)
        with self.test_client() as c:
            r = c.get('/v3/users/%s/access_rules' % self.user_id, headers=self.headers)
            self.assertEqual(len(r.json['access_rules']), 1)

    def test_user_can_delete_their_access_rules(self):
        access_rule_id = uuid.uuid4().hex
        app_cred = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'user_id': self.user_id, 'project_id': self.project_id, 'secret': uuid.uuid4().hex, 'access_rules': [{'id': access_rule_id, 'service': uuid.uuid4().hex, 'path': uuid.uuid4().hex, 'method': uuid.uuid4().hex[16:]}]}
        PROVIDERS.application_credential_api.create_application_credential(app_cred)
        PROVIDERS.application_credential_api.delete_application_credential(app_cred['id'])
        with self.test_client() as c:
            path = '/v3/users/%s/access_rules/%s' % (self.user_id, access_rule_id)
            c.delete(path, headers=self.headers)