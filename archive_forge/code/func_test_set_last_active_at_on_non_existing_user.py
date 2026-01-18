import datetime
from unittest import mock
import uuid
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import sql_model as model
from keystone.identity.shadow_backends import sql as shadow_sql
from keystone.tests import unit
def test_set_last_active_at_on_non_existing_user(self):
    self.config_fixture.config(group='security_compliance', disable_user_account_days_inactive=90)
    password = uuid.uuid4().hex
    user = self._create_user(password)
    real_last_active_at = shadow_sql.ShadowUsers.set_last_active_at
    test_self = self

    def fake_last_active_at(self, user_id):
        test_self._delete_user(user_id)
        real_last_active_at(self, user_id)
    with mock.patch.object(shadow_sql.ShadowUsers, 'set_last_active_at', fake_last_active_at):
        with self.make_request():
            PROVIDERS.identity_api.authenticate(user_id=user['id'], password=password)