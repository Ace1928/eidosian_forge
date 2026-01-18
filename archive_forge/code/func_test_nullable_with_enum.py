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
def test_nullable_with_enum(self):
    schema_with_enum = {'type': 'object', 'properties': {'test': validation.nullable(parameter_types.boolean)}, 'additionalProperties': False, 'required': ['test']}
    self.assertIn('null', schema_with_enum['properties']['test']['type'])
    self.assertIn(None, schema_with_enum['properties']['test']['enum'])
    validator = validators.SchemaValidator(schema_with_enum)
    reqs_to_validate = [{'test': val} for val in [True, False, None]]
    for req in reqs_to_validate:
        validator.validate(req)