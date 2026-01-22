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
class TestMappingLocals(unit.BaseTestCase):
    mapping_split = {'rules': [{'local': [{'user': {'name': '{0}'}}, {'group': {'id': 'd34db33f'}}], 'remote': [{'type': 'idp_username'}]}]}
    mapping_combined = {'rules': [{'local': [{'user': {'name': '{0}'}, 'group': {'id': 'd34db33f'}}], 'remote': [{'type': 'idp_username'}]}]}
    mapping_with_duplicate = {'rules': [{'local': [{'user': {'name': 'test_{0}'}}, {'user': {'name': '{0}'}}], 'remote': [{'type': 'idp_username'}]}]}
    assertion = {'idp_username': 'a_user'}

    def process(self, rules):
        rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, rules)
        return rp.process(self.assertion)

    def test_local_list_gets_squashed_into_a_single_dictionary(self):
        expected = {'user': {'name': 'a_user', 'type': 'ephemeral'}, 'projects': [], 'group_ids': ['d34db33f'], 'group_names': []}
        mapped_split = self.process(self.mapping_split['rules'])
        mapped_combined = self.process(self.mapping_combined['rules'])
        self.assertEqual(expected, mapped_split)
        self.assertEqual(mapped_split, mapped_combined)

    def test_when_local_list_gets_squashed_first_dict_wins(self):
        expected = {'user': {'name': 'test_a_user', 'type': 'ephemeral'}, 'projects': [], 'group_ids': [], 'group_names': []}
        mapped = self.process(self.mapping_with_duplicate['rules'])
        self.assertEqual(expected, mapped)