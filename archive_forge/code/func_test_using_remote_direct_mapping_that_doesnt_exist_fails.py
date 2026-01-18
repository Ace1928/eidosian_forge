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
def test_using_remote_direct_mapping_that_doesnt_exist_fails(self):
    """Test for the correct error when referring to a bad remote match.

        The remote match must exist in a rule when a local section refers to
        a remote matching using the format (e.g. {0} in a local section).
        """
    mapping = mapping_fixtures.MAPPING_DIRECT_MAPPING_THROUGH_KEYWORD
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.CUSTOMER_ASSERTION
    self.assertRaises(exception.DirectMappingError, rp.process, assertion)