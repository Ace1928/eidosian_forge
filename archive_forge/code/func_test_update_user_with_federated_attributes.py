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
def test_update_user_with_federated_attributes(self):
    """Call ``PATCH /users/{user_id}``."""
    user = self.fed_user.copy()
    del user['id']
    user['name'] = 'James Doe'
    idp, protocol = self._create_federated_attributes()
    user['federated'] = [{'idp_id': idp['id'], 'protocols': [{'protocol_id': protocol['id'], 'unique_id': 'jdoe'}]}]
    r = self.patch('/users/%(user_id)s' % {'user_id': self.fed_user['id']}, body={'user': user})
    resp_user = r.result['user']
    self.assertEqual(user['name'], resp_user['name'])
    self.assertEqual(user['federated'], resp_user['federated'])
    self.assertValidUserResponse(r, user)