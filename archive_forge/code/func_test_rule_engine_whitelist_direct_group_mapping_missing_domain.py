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
def test_rule_engine_whitelist_direct_group_mapping_missing_domain(self):
    """Test if the local rule is rejected upon missing domain value.

        This is a variation with a ``whitelist`` filter.

        """
    mapping = mapping_fixtures.MAPPING_GROUPS_WHITELIST_MISSING_DOMAIN
    assertion = mapping_fixtures.EMPLOYEE_ASSERTION_MULTIPLE_GROUPS
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    self.assertRaises(exception.ValidationError, rp.process, assertion)