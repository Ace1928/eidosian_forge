import datetime
from unittest import mock
import uuid
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import revoke_model
from keystone.revoke.backends import sql
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_backend_sql
from keystone.token import provider
class FernetSqlRevokeTests(test_backend_sql.SqlTests, RevokeTests):

    def config_overrides(self):
        super(FernetSqlRevokeTests, self).config_overrides()
        self.config_fixture.config(group='token', provider='fernet', revoke_by_id=False)
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))