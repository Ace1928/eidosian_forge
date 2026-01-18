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
def test_rule_engine_returns_group_names(self):
    """Check whether RuleProcessor returns group names with their domains.

        RuleProcessor should return 'group_names' entry with a list of
        dictionaries with two entries 'name' and 'domain' identifying group by
        its name and domain.

        """
    mapping = mapping_fixtures.MAPPING_GROUP_NAMES
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.EMPLOYEE_ASSERTION
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertValidMappedUserObject(mapped_properties)
    reference = {mapping_fixtures.DEVELOPER_GROUP_NAME: {'name': mapping_fixtures.DEVELOPER_GROUP_NAME, 'domain': {'name': mapping_fixtures.DEVELOPER_GROUP_DOMAIN_NAME}}, mapping_fixtures.TESTER_GROUP_NAME: {'name': mapping_fixtures.TESTER_GROUP_NAME, 'domain': {'id': mapping_fixtures.DEVELOPER_GROUP_DOMAIN_ID}}}
    for rule in mapped_properties['group_names']:
        self.assertDictEqual(reference.get(rule.get('name')), rule)