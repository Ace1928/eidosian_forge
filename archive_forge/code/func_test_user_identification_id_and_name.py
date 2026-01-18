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
def test_user_identification_id_and_name(self):
    """Test various mapping options and how users are identified.

        This test calls mapped.setup_username() for propagating user object.

        Test plan:
        - Check if the user has proper domain ('federated') set
        - Check if the user has proper type set ('ephemeral')
        - Check if display_name is properly set from the assertion
        - Check if unique_id is properly set and equal to value hardcoded
        in the mapping

        This test does two iterations with different assertions used as input
        for the Mapping Engine.  Different assertions will be matched with
        different rules in the ruleset, effectively issuing different user_id
        (hardcoded values). In the first iteration, the hardcoded user_id is
        not url-safe and we expect Keystone to make it url safe. In the latter
        iteration, provided user_id is already url-safe and we expect server
        not to change it.

        """
    testcases = [(mapping_fixtures.CUSTOMER_ASSERTION, 'bwilliams'), (mapping_fixtures.EMPLOYEE_ASSERTION, 'tbo')]
    for assertion, exp_user_name in testcases:
        mapping = mapping_fixtures.MAPPING_USER_IDS
        rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
        mapped_properties = rp.process(assertion)
        self.assertIsNotNone(mapped_properties)
        self.assertValidMappedUserObject(mapped_properties)
        resource_api_mock = mock.patch('keystone.resource.core.DomainConfigManager')
        idp_domain_id = uuid.uuid4().hex
        user_domain_id = mapped_properties['user']['domain']['id']
        mapped.validate_and_prepare_federated_user(mapped_properties, idp_domain_id, resource_api_mock)
        self.assertEqual(exp_user_name, mapped_properties['user']['name'])
        self.assertEqual('abc123%40example.com', mapped_properties['user']['id'])
        self.assertEqual(user_domain_id, mapped_properties['user']['domain']['id'])