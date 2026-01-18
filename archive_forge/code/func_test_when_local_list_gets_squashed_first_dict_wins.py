import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
def test_when_local_list_gets_squashed_first_dict_wins(self):
    expected = {'user': {'name': 'test_a_user', 'type': 'ephemeral'}, 'projects': [], 'group_ids': [], 'group_names': []}
    mapped = self.process(self.mapping_with_duplicate['rules'])
    self.assertEqual(expected, mapped)