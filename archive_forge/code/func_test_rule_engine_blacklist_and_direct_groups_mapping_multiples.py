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
def test_rule_engine_blacklist_and_direct_groups_mapping_multiples(self):
    """Test matching multiple values before the blacklist.

        Verifies that the local indexes are correct when matching multiple
        remote values for a field when the field occurs before the blacklist
        entry in the remote rules.

        """
    mapping = mapping_fixtures.MAPPING_GROUPS_BLACKLIST_MULTIPLES
    assertion = mapping_fixtures.EMPLOYEE_ASSERTION_MULTIPLE_GROUPS
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    reference = {mapping_fixtures.CONTRACTOR_GROUP_NAME: {'name': mapping_fixtures.CONTRACTOR_GROUP_NAME, 'domain': {'id': mapping_fixtures.DEVELOPER_GROUP_DOMAIN_ID}}}
    for rule in mapped_properties['group_names']:
        self.assertDictEqual(reference.get(rule.get('name')), rule)
    self.assertEqual('tbo', mapped_properties['user']['name'])
    self.assertEqual([], mapped_properties['group_ids'])