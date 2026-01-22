import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def _create_user_with_expired_password(self):
    expire_days = CONF.security_compliance.password_expires_days + 1
    time = datetime.datetime.utcnow() - datetime.timedelta(expire_days)
    password = uuid.uuid4().hex
    user_ref = unit.new_user_ref(domain_id=self.domain_id, password=password)
    with freezegun.freeze_time(time):
        self.user_ref = PROVIDERS.identity_api.create_user(user_ref)
    return password