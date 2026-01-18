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
def test_rule_engine_regex_many_groups(self):
    """Should return group CONTRACTOR_GROUP_ID.

        The TESTER_ASSERTION should successfully have a match in
        MAPPING_TESTER_REGEX. This will test the case where many groups
        are in the assertion, and a regex value is used to try and find
        a match.

        """
    mapping = mapping_fixtures.MAPPING_TESTER_REGEX
    assertion = mapping_fixtures.TESTER_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    values = rp.process(assertion)
    self.assertValidMappedUserObject(values)
    user_name = assertion.get('UserName')
    group_ids = values.get('group_ids')
    name = values.get('user', {}).get('name')
    self.assertEqual(user_name, name)
    self.assertIn(mapping_fixtures.TESTER_GROUP_ID, group_ids)