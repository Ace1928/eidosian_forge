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
def test_revoke_by_audit_id(self):
    token = _sample_blank_token()
    token['audit_id'] = uuid.uuid4().hex
    token['audit_chain_id'] = token['audit_id']
    PROVIDERS.revoke_api.revoke_by_audit_id(audit_id=token['audit_id'])
    self._assertTokenRevoked(token)
    token2 = _sample_blank_token()
    token2['audit_id'] = uuid.uuid4().hex
    token2['audit_chain_id'] = token2['audit_id']
    self._assertTokenNotRevoked(token2)