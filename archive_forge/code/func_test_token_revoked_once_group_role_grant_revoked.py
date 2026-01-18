import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_token_revoked_once_group_role_grant_revoked(self):
    """Test token invalid when direct & indirect role on user is revoked.

        When a role granted to a group is revoked for a given scope,
        and user direct role is revoked, then tokens created
        by user will be invalid.

        """
    time = datetime.datetime.utcnow()
    with freezegun.freeze_time(time) as frozen_datetime:
        PROVIDERS.assignment_api.create_grant(role_id=self.role['id'], project_id=self.project['id'], group_id=self.group['id'])
        PROVIDERS.identity_api.add_user_to_group(user_id=self.user['id'], group_id=self.group['id'])
        auth_body = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        token_resp = self.post('/auth/tokens', body=auth_body)
        token = token_resp.headers.get('x-subject-token')
        self.head('/auth/tokens', headers={'x-subject-token': token}, expected_status=http.client.OK)
        frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
        PROVIDERS.assignment_api.delete_grant(role_id=self.role['id'], project_id=self.project['id'], group_id=self.group['id'])
        PROVIDERS.assignment_api.delete_grant(role_id=self.role['id'], project_id=self.project['id'], user_id=self.user['id'])
        frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
        self.head('/auth/tokens', token=token, expected_status=http.client.UNAUTHORIZED)