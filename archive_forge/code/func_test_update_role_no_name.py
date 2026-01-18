from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_update_role_no_name(self):
    PROVIDERS.role_api.update_role(self.role_member['id'], {'description': uuid.uuid4().hex})