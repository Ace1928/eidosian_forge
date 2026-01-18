import copy
from unittest import mock
import uuid
from oslo_config import cfg
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_option_dict_returns_valid_config_values(self):
    regex = uuid.uuid4().hex
    self.config_fixture.config(group='security_compliance', password_regex=regex)
    expected_dict = {'group': 'security_compliance', 'option': 'password_regex', 'value': regex}
    option_dict = PROVIDERS.domain_config_api._option_dict('security_compliance', 'password_regex')
    self.assertEqual(option_dict, expected_dict)