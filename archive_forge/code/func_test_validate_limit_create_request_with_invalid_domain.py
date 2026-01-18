import copy
import uuid
from keystone.application_credential import schema as app_cred_schema
from keystone.assignment import schema as assignment_schema
from keystone.catalog import schema as catalog_schema
from keystone.common import validation
from keystone.common.validation import parameter_types
from keystone.common.validation import validators
from keystone.credential import schema as credential_schema
from keystone import exception
from keystone.federation import schema as federation_schema
from keystone.identity.backends import resource_options as ro
from keystone.identity import schema as identity_schema
from keystone.limit import schema as limit_schema
from keystone.oauth1 import schema as oauth1_schema
from keystone.policy import schema as policy_schema
from keystone.resource import schema as resource_schema
from keystone.tests import unit
from keystone.trust import schema as trust_schema
def test_validate_limit_create_request_with_invalid_domain(self):
    request_to_validate = [{'domain_id': 'fake_id', 'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'resource_limit': 10, 'description': 'test description'}]
    self.assertRaises(exception.SchemaValidationError, self.create_limits_validator.validate, request_to_validate)