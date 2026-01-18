import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_v3_authenticate_user_name_and_domain_id(self):
    user_id = self.user_id
    user_name = self.user['name']
    password = self.user['password']
    domain_id = self.domain_id
    data = self.build_authentication_request(username=user_name, user_domain_id=domain_id, password=password)
    self.post('/auth/tokens', body=data)
    self._assert_last_note(self.ACTION, user_id)