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
def test_rule_engine_group_ids_mapping_whitelist(self):
    """Test mapping engine when group_ids is explicitly set.

        Also test whitelists on group ids

        """
    mapping = mapping_fixtures.MAPPING_GROUPS_IDS_WHITELIST
    assertion = mapping_fixtures.GROUP_IDS_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertEqual('opilotte', mapped_properties['user']['name'])
    self.assertListEqual([], mapped_properties['group_names'])
    self.assertCountEqual(['abc123', 'ghi789', 'klm012'], mapped_properties['group_ids'])