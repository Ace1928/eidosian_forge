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
def test_rule_engine_not_any_of_regex_verify_pass(self):
    """Should return group DEVELOPER_GROUP_ID.

        The DEVELOPER_ASSERTION should successfully have a match in
        MAPPING_DEVELOPER_REGEX. This will test the case where many
        remote rules must be matched, including a `not_any_of`, with
        regex set to True.

        """
    mapping = mapping_fixtures.MAPPING_DEVELOPER_REGEX
    assertion = mapping_fixtures.DEVELOPER_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    values = rp.process(assertion)
    self.assertValidMappedUserObject(values)
    user_name = assertion.get('UserName')
    group_ids = values.get('group_ids')
    name = values.get('user', {}).get('name')
    self.assertEqual(user_name, name)
    self.assertIn(mapping_fixtures.DEVELOPER_GROUP_ID, group_ids)