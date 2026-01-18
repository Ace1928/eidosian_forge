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
def test_rule_engine_any_one_of_and_direct_mapping(self):
    """Should return user's name and group id EMPLOYEE_GROUP_ID.

        The ADMIN_ASSERTION should successfully have a match in MAPPING_LARGE.
        They will test the case where `any_one_of` is valid, and there is
        a direct mapping for the users name.

        """
    mapping = mapping_fixtures.MAPPING_LARGE
    assertion = mapping_fixtures.ADMIN_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    values = rp.process(assertion)
    fn = assertion.get('FirstName')
    ln = assertion.get('LastName')
    full_name = '%s %s' % (fn, ln)
    group_ids = values.get('group_ids')
    user_name = values.get('user', {}).get('name')
    self.assertIn(mapping_fixtures.EMPLOYEE_GROUP_ID, group_ids)
    self.assertEqual(full_name, user_name)