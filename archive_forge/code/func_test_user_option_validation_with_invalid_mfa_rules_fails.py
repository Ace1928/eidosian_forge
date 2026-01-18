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
def test_user_option_validation_with_invalid_mfa_rules_fails(self):
    test_cases = [(True, TypeError), ([True, False], TypeError), ([[True], [True, False]], TypeError), ([['duplicate_array'] for x in range(0, 2)], ValueError), ([[uuid.uuid4().hex], []], ValueError), ([['duplicate' for x in range(0, 2)]], ValueError)]
    for ruleset, exception_class in test_cases:
        request_to_validate = {'options': {ro.MFA_RULES_OPT.option_name: ruleset}}
        self.assertRaises(exception.SchemaValidationError, self.update_user_validator.validate, request_to_validate)
        self.assertRaises(exception_class, ro._mfa_rules_validator_list_of_lists_of_strings_no_duplicates, ruleset)