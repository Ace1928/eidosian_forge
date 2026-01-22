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
def _user_field_test(self, field_name):
    token = _sample_blank_token()
    token[field_name] = uuid.uuid4().hex
    PROVIDERS.revoke_api.revoke_by_user(user_id=token[field_name])
    self._assertTokenRevoked(token)
    token2 = _sample_blank_token()
    token2[field_name] = uuid.uuid4().hex
    self._assertTokenNotRevoked(token2)