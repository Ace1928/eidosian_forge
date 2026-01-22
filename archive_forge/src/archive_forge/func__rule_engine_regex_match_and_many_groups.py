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
def _rule_engine_regex_match_and_many_groups(self, assertion):
    """Should return group DEVELOPER_GROUP_ID and TESTER_GROUP_ID.

        A helper function injecting assertion passed as an argument.
        Expect DEVELOPER_GROUP_ID and TESTER_GROUP_ID in the results.

        """
    mapping = mapping_fixtures.MAPPING_LARGE
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    values = rp.process(assertion)
    user_name = assertion.get('UserName')
    group_ids = values.get('group_ids')
    name = values.get('user', {}).get('name')
    self.assertValidMappedUserObject(values)
    self.assertEqual(user_name, name)
    self.assertIn(mapping_fixtures.DEVELOPER_GROUP_ID, group_ids)
    self.assertIn(mapping_fixtures.TESTER_GROUP_ID, group_ids)